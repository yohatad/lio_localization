# ICP localization against a prior .pcd map, on POINT-LIO odometry -- the
# Point-LIO equivalent of fastlio_localization_l2.launch.py. See that file for
# the reasoning behind each piece; this documents only what differs.
#
# WHY A REMAP IS ENOUGH (no C++ changes):
#   * /cloud_registered  -- Point-LIO publishes this under the IDENTICAL name,
#     in the same 'odom' frame (its odom_header_frame_id), so global_localization
#     consumes it unchanged.
#   * /Odometry          -- hardcoded in BOTH nodes (global_localization.cpp and
#     transform_fusion.cpp), and Point-LIO publishes /aft_mapped_to_init instead.
#     A launch-time remap on both nodes covers it. Remapping (not editing the
#     source) keeps this package usable by either backend at once.
#
# THE PRIOR MAP IS BACKEND-AGNOSTIC. map_pcd is just environment geometry, so a
# map built by the FAST-LIO PGO run localizes Point-LIO perfectly well. The
# default therefore points at the SAME map as the FAST-LIO variant rather than
# at pointlio_lc_l2.launch.py's separate save_directory -- one map, either
# backend. Pass map_pcd:= to use a Point-LIO-built map instead.
#
# Usage:
#   ros2 launch lio_localization pointlio_localization_l2.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data
#   Then give it an initial guess in RViz (2D Pose Estimate) -- global
#   localization needs a rough seed before ICP can converge.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Point-LIO's odometry topic. Both localization nodes hardcode "/Odometry"
# (FAST-LIO's name), so every Node below remaps it to this.
POINT_LIO_ODOM_TOPIC = '/aft_mapped_to_init'


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
            f"against a PRE-BUILT map -- build one with fastlio_lc_l2.launch.py "
            f"or pointlio_lc_l2.launch.py (each writes map_batch.pcd under its "
            f"save_directory once you call 'ros2 service call "
            f"/pgo_batch_optimize std_srvs/srv/Trigger'), or pass map_pcd:=<path> "
            f"to point at an existing map.")
    return []


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
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
        description='Prior map .pcd to localize against (the loop-closed PGO map). '
                    'Backend-agnostic geometry -- a FAST-LIO-built map is fine here.'
    )
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    # A LOCALIZATION view (Fixed Frame 'map', submap vs scan overlap), not
    # point_lio's mapping config -- that one is fixed to 'odom', where every ICP
    # correction moves the map instead of the robot, and its /Odometry display
    # is dead here anyway since Point-LIO publishes /aft_mapped_to_init.
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=os.path.join(
            localization_share, 'rviz', 'localization.rviz'))
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot.')
    declare_localization_th_cmd = DeclareLaunchArgument(
        'localization_th', default_value='0.90',
        description='Min ICP inlier-ratio fitness to accept a match. Lower if the '
                    'L2 scan only partly overlaps the prior map (0.6-0.9 typical).')

    # base_footprint -> l2lidar_frame -> l2lidar_frame_imu (+ cams). The lio bridge
    # needs the static base_footprint -> l2lidar_frame_imu to close odom->base_footprint.
    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensor_tf_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # Point-LIO odometry ONLY (no PGO). Publishes /aft_mapped_to_init +
    # /cloud_registered in the 'odom_lidar' frame; lio_map_odom_bridge closes
    # odom_lidar -> base_footprint. 'odom' (leveled) is published as a CHILD of
    # odom_lidar: transform_fusion owns map -> odom_lidar so it cannot also be
    # odom_lidar's parent, but the costmaps and collision monitor in
    # nav2_params_fastlio_loc.yaml require odom (raw odom_lidar is
    # backend-native and not gravity-aligned).
    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch',
                         'mapping_l2lidar_node.launch.py')),
        launch_arguments={
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            'bridge_level_frame': 'true',
            'level_frame_as_child': 'true',
        }.items())

    # Registers /cloud_registered (odom frame) against the prior map, producing map -> odom_lidar.
    global_localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='point_lio_localization',
        output='screen',
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_pcd': map_pcd,
            'map_frame': 'map',
            'odom_frame': 'odom_lidar',
            'map_voxel_size': 0.4,
            'scan_voxel_size': 0.1,
            'freq_localization': 0.5,
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
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
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
    ld.add_action(OpaqueFunction(function=_check_map_pcd_exists))
    ld.add_action(sensor_tf_launch)
    ld.add_action(point_lio_launch)
    ld.add_action(global_localization)
    ld.add_action(transform_fusion)
    return ld
