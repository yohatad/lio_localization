# Prior-map ICP localization on POINT-LIO odometry -- the Point-LIO twin of
# fastlio_localization_l2.launch.py. That file carries the shared reasoning;
# this one documents only what differs:
#
#   * A remap is all it takes. Point-LIO publishes /cloud_registered under the
#     identical name and frame, and both localization nodes hardcode
#     "/Odometry", so remapping that to Point-LIO's topic keeps this package
#     usable by either backend without touching the C++.
#   * The prior map is backend-agnostic geometry, so map_pcd and keyframe_poses
#     default to the SAME artifacts as the FAST-LIO variant. Pass map_pcd:= to
#     use a Point-LIO-built map instead.
#
# Runtime tunables live in config/localization.yaml. Seeding: publish
# /initialpose, call /relocalize, or pass auto_initialize:=true.

import os

from ament_index_python.packages import (
    PackageNotFoundError, get_package_share_directory)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# mapping_l2lidar_node.launch.py remaps Point-LIO's native /aft_mapped_to_init
# to this shared name; the nodes below hardcode "/Odometry", so remap to it.
POINT_LIO_ODOM_TOPIC = '/odom_lio'


def _check_map_pcd_exists(context, *args, **kwargs):
    """Fail at launch with an actionable message, not in the node's C++ ctor."""
    path = LaunchConfiguration('map_pcd').perform(context)
    if not os.path.isfile(path):
        raise RuntimeError(
            f"map_pcd '{path}' does not exist. This launch file localizes "
            f"against a PRE-BUILT map -- build one with fastlio_lc_l2.launch.py "
            f"or pointlio_lc_l2.launch.py (each writes map_batch.pcd under its "
            f"save_directory once you call 'ros2 service call "
            f"/pgo_batch_optimize std_srvs/srv/Trigger'), or pass map_pcd:=<path> "
            f"to point at an existing map.")
    return []


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
    sensor_tf_share = get_package_share_directory('pepper_slam')
    # pepper_navigation ships the map artifacts but is deliberately NOT an
    # exec_depend: it already depends on this package, and the reverse edge
    # makes colcon refuse the workspace. Tolerate its absence so this package
    # still launches standalone.
    try:
        nav_share = get_package_share_directory('pepper_navigation')
    except PackageNotFoundError:
        nav_share = ''
    localization_share = get_package_share_directory('lio_localization')

    map_pcd = LaunchConfiguration('map_pcd')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value=os.path.join(nav_share, 'pcd', 'pepper_map_lc.pcd'),
        description='Prior map .pcd to localize against (the loop-closed PGO map). '
                    'Backend-agnostic geometry -- a FAST-LIO-built map is fine here.'
    )
    # Enables SEEDLESS localization: the ICP is purely local, so without a
    # candidate set a seed more than a few metres out can never recover.
    declare_kf_poses_cmd = DeclareLaunchArgument(
        'keyframe_poses',
        default_value=os.path.join(nav_share, 'pcd', 'pepper_map_lc_poses.txt'),
        description='KITTI-format keyframe poses from the mapping run, used as '
                    'candidates for global localization and /relocalize. Empty '
                    'disables both (manual /initialpose only). Must come from '
                    'the SAME run as map_pcd; backend-agnostic like map_pcd.'
    )
    declare_auto_init_cmd = DeclareLaunchArgument(
        'auto_initialize', default_value='false',
        description='Run the global search automatically at startup instead of '
                    'waiting for /initialpose or /relocalize. Off by default: a '
                    'silent lock onto the wrong place is worse than waiting.')
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config under point_lio/config. l2lidar_rsimu.yaml '
                    'drives the estimator from the RealSense IMU (default); '
                    'l2lidar_node.yaml uses the L2 s own, which loses heading in '
                    'slow turns (utils/L2_IMU/REPORT.md). Pair with '
                    'lidar_imu_frame.')
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Static frame the estimated body corresponds to. '
                    'camera_imu_optical_frame for l2lidar_rsimu.yaml, '
                    'l2lidar_frame_imu for l2lidar_node.yaml.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        # A LOCALIZATION view (fixed frame 'map'), not Point-LIO's mapping
        # config -- that one is fixed to 'odom', where every ICP correction
        # moves the map around a stationary robot.
        default_value=os.path.join(
            localization_share, 'rviz', 'localization.rviz'))
    # false, NOT true: this is the LIVE entry point, and 'true' on the robot
    # pins sim time at 0, so tf never resolves and nothing fuses, silently.
    # pepper_sensor_tf's publisher/scope are NOT derived from this -- on a bag
    # pass publisher:=none if it carries its own /tf_static, publisher:=urdf
    # scope:=all if it does not.
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='false (default) on the robot; true for bag replay with '
                    'ros2 bag play --clock. The bag_test wrappers set this.')
    declare_params_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(localization_share, 'config', 'localization.yaml'),
        description='YAML of runtime tunables. Point this at the SOURCE copy '
                    '(src/lio_localization/config/localization.yaml) to edit and '
                    'relaunch without rebuilding.')

    # Static rig TF. The bridge needs base_footprint -> l2lidar_frame_imu from
    # here to close odom -> base_footprint.
    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensor_tf_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # Point-LIO odometry ONLY (no PGO), in the 'lio_init' frame; its own launch
    # starts lio_odom_bridge. 'odom' (leveled) is a CHILD of lio_init:
    # transform_fusion owns map -> lio_init, so it cannot also be lio_init's
    # parent -- but the costmaps need a gravity-aligned frame and raw lio_init
    # is not one.
    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch',
                         'mapping_l2lidar_node.launch.py')),
        launch_arguments={
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            'config_file': LaunchConfiguration('config_file'),
            'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
            'bridge_level_frame': 'true',
            'level_frame_as_child': 'true',
        }.items())

    # Registers /cloud_registered against the prior map -> map -> lio_init.
    #
    # In both nodes below: params_file FIRST, then ONLY values that must come
    # from launch context. Anything named in both is won by the dict, which
    # silently makes that key inert in the YAML -- so keep the dict minimal.
    global_localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='point_lio_localization',
        output='screen',
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': use_sim_time,
            'map_pcd': map_pcd,
            'keyframe_poses': LaunchConfiguration('keyframe_poses'),
            'auto_initialize': ParameterValue(
                LaunchConfiguration('auto_initialize'), value_type=bool),
            'map_frame': 'map',
            'odom_frame': 'lio_init',
            'fov': 6.28,        # L2 is 360 deg -> ring crop (distance only)
        }],
    )

    # Broadcasts map -> lio_init from the latest correction.
    transform_fusion = Node(
        package='lio_localization',
        executable='transform_fusion',
        name='transform_fusion',
        output='screen',
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': use_sim_time,
            'map_frame': 'map',
            'odom_frame': 'lio_init',
            'body_frame': 'base_footprint',   # /localization pose is map -> base
        }],
    )

    ld = LaunchDescription()
    ld.add_action(declare_map_pcd_cmd)
    ld.add_action(declare_kf_poses_cmd)
    ld.add_action(declare_auto_init_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_lidar_imu_frame_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_cmd)
    ld.add_action(OpaqueFunction(function=_check_map_pcd_exists))
    ld.add_action(sensor_tf_launch)
    ld.add_action(point_lio_launch)
    ld.add_action(global_localization)
    ld.add_action(transform_fusion)
    return ld
