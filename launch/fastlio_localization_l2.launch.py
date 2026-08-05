import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _check_map_pcd_exists(context, *args, **kwargs):
    """Fail at launch time, not deep in global_localization's C++ ctor.

    This is an expected precondition (you need a prior map before you can
    localize against one), not a bug -- but failing here gives a much more
    actionable message than the node's runtime exception, including how to
    actually build the map this launch file expects.
    """
    path = LaunchConfiguration('map_pcd').perform(context)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"map_pcd '{path}' does not exist. This launch file localizes "
            f"against a PRE-BUILT map -- build one first with "
            f"fastlio_lc_l2.launch.py (it writes map_batch.pcd under its "
            f"save_directory once PGO finishes), or pass map_pcd:=<path> to "
            f"point at an existing map.")
    return []


def generate_launch_description():
    fast_lio_share = get_package_share_directory('fast_lio')
    sensor_tf_share = get_package_share_directory('pepper_slam')
    localization_share = get_package_share_directory('lio_localization')

    map_pcd = LaunchConfiguration('map_pcd')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    localization_th = LaunchConfiguration('localization_th')

    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value='/home/yoha/Lidar/run_l2_lc/pgo_output/map_batch.pcd',
        description='Prior map .pcd to localize against (the loop-closed PGO map).'
    )
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        # A LOCALIZATION view (Fixed Frame 'map', submap vs scan overlap), not
        # FAST-LIO's mapping config -- that one is fixed to 'odom', where every
        # ICP correction moves the map around a stationary robot.
        default_value=os.path.join(
            localization_share, 'rviz', 'localization.rviz'))
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot.')
    declare_localization_th_cmd = DeclareLaunchArgument(
        'localization_th', default_value='0.70',
        description='Min ICP inlier-ratio fitness to accept a match. 0.90 was '
                    'upstream\'s value and is too strict for the L2 on this rig: '
                    'measured 0.793 on bags/July_22, so EVERY match was rejected '
                    'and no map -> odom_lidar was ever published. At 0.70 the same '
                    'bag locks at 0.978. Raise it if you see it locking onto the '
                    'wrong place; lower it further if it never locks at all.')
    declare_freq_cmd = DeclareLaunchArgument(
        'freq_localization', default_value='2.0',
        description='ICP corrections per second. Upstream shipped 0.5 Hz, which '
                    'leaves the pose frozen for 2 s at a time -- on a moving '
                    'robot the next ICP then starts from a stale seed. Each '
                    'match measured 25-75 ms here, so 2 Hz is ~10% duty cycle '
                    'and 5 Hz is still comfortable. Raise for faster motion, '
                    'lower if CPU-bound.')

    # base_footprint -> l2lidar_frame -> l2lidar_frame_imu (+ cams). The lio bridge
    # needs the static base_footprint -> l2lidar_frame_imu to close odom->base_footprint.
    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensor_tf_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # FAST-LIO odometry ONLY (no PGO). Publishes /Odometry + /cloud_registered
    # in the 'odom_lidar' frame; lio_map_odom_bridge closes
    # odom_lidar -> base_footprint.
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={
            'config_file': 'l2.yaml',
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            # 'odom' (leveled) is published as a CHILD of odom_lidar here:
            # transform_fusion owns map -> odom_lidar, so it cannot also be
            # odom_lidar's parent. Leveling stays ON because the costmaps and
            # collision monitor need a gravity-aligned, floor-referenced frame,
            # and raw odom_lidar is neither (and means different things on
            # FAST-LIO vs Point-LIO).
            'bridge_level_frame': 'true',
            'level_frame_as_child': 'true',
        }.items())

    # Registers /cloud_registered (odom frame) against the prior map, producing map -> odom_lidar.
    global_localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='fast_lio_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_pcd': map_pcd,
            'map_frame': 'map',
            'odom_frame': 'odom_lidar',
            'map_voxel_size': 0.4,
            'scan_voxel_size': 0.1,
            'freq_localization': ParameterValue(
                LaunchConfiguration('freq_localization'), value_type=float),
            'localization_th': ParameterValue(localization_th, value_type=float),
            'fov': 6.28,        # L2 is 360 deg -> ring crop (distance only)
            'fov_far': 30.0,
        }],
    )

    # Broadcasts map -> odom TF at 50 Hz from the latest correction.
    transform_fusion = Node(
        package='lio_localization',
        executable='transform_fusion',
        name='transform_fusion',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_frame': 'map',
            'odom_frame': 'odom_lidar',
            'body_frame': 'base_footprint',   # /localization pose is map -> base
            'fusion_rate': 50.0,
        }],
    )

    ld = LaunchDescription()
    ld.add_action(declare_map_pcd_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_localization_th_cmd)
    ld.add_action(declare_freq_cmd)
    ld.add_action(OpaqueFunction(function=_check_map_pcd_exists))
    ld.add_action(sensor_tf_launch)
    ld.add_action(fast_lio_launch)
    ld.add_action(global_localization)
    ld.add_action(transform_fusion)
    return ld
