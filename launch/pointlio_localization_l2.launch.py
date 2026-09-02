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

from ament_index_python.packages import (
    PackageNotFoundError, get_package_share_directory)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Point-LIO's odometry topic. Both localization nodes hardcode "/Odometry"
# (FAST-LIO's name), so every Node below remaps it to this.
# Point-LIO now remaps its native /aft_mapped_to_init to the shared /odom_lio
# in mapping_l2lidar_node.launch.py, so localization nodes -- which
# hardcode "/Odometry" -- are remapped to that instead.
POINT_LIO_ODOM_TOPIC = '/odom_lio'


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
    # Map artifacts ship with pepper_navigation so map_pcd and keyframe_poses
    # resolve on ANY machine (the Jetson runs as a different user, so an
    # absolute /home/<user> default silently does not exist there).
    #
    # Deliberately NOT declared as an exec_depend in package.xml: pepper_navigation
    # already depends on THIS package, so declaring the reverse edge makes colcon
    # refuse the whole workspace with "Unable to order packages topologically".
    # The ament index lookup does not need the declaration -- it only needs the
    # package to be built. Tolerate its absence so this package still launches
    # standalone; the defaults are then unusable placeholders, and callers
    # (pepper_nav2_fastlio_loc.launch.py) pass both paths explicitly anyway.
    try:
        nav_share = get_package_share_directory('pepper_navigation')
    except PackageNotFoundError:
        nav_share = ''
    localization_share = get_package_share_directory('lio_localization')

    map_pcd = LaunchConfiguration('map_pcd')
    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    localization_th = LaunchConfiguration('localization_th')

    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value=os.path.join(nav_share, 'pcd', 'pepper_map_lc.pcd'),
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
    # false, NOT true: this is the LIVE entry point. Every wrapper in
    # pepper_slam/launch/bag_test sets use_sim_time:='true' explicitly, so this
    # default only ever applies on the robot -- where 'true' pins sim time at 0,
    # so tf never resolves and nothing fuses, silently and with no error.
    # It also feeds the sensor_tf scope derivation: false -> 'mount', correct
    # live because the RealSense driver publishes its own camera edges (adding a
    # second copy is the nondeterministic-latch problem sensor_tf.yaml warns of).
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='false (default) on the robot; true for bag replay with ros2 bag play --clock. The bag_test wrappers set this for you.')
    declare_localization_th_cmd = DeclareLaunchArgument(
        'localization_th', default_value='0.95',
        description='Min ICP inlier-ratio fitness to accept a match. 0.90 was '
                    'upstream\'s value and is too strict for the L2 on this rig: '
                    'measured 0.793 on bags/July_22, so EVERY match was rejected '
                    'and no map -> lio_init was ever published. At 0.70 the same '
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

    # Point-LIO odometry ONLY (no PGO). Publishes /aft_mapped_to_init +
    # /cloud_registered in the 'lio_init' frame; lio_odom_bridge closes
    # lio_init -> base_footprint. 'odom' (leveled) is published as a CHILD of
    # lio_init: transform_fusion owns map -> lio_init so it cannot also be
    # lio_init's parent, but the costmaps and collision monitor in
    # nav2_params_fastlio_loc.yaml require odom (raw lio_init is
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

    # Registers /cloud_registered (odom frame) against the prior map, producing map -> lio_init.
    global_localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='point_lio_localization',
        output='screen',
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
        # params_file FIRST, then only the values that must come from launch
        # context. Anything appearing in BOTH would be won by this dict, which
        # silently made the YAML inert for every tunable in it.
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': use_sim_time,
            'map_pcd': map_pcd,
            'map_frame': 'map',
            'odom_frame': 'lio_init',
            'scan_voxel_size': 0.1,
            'fov': 6.28,        # L2 is 360 deg -> ring crop (distance only)
        }],
    )

    # Broadcasts map -> odom TF at 50 Hz from the latest correction.
    transform_fusion = Node(
        package='lio_localization',
        executable='transform_fusion',
        name='transform_fusion',
        output='screen',
        remappings=[('/Odometry', POINT_LIO_ODOM_TOPIC)],
        # params_file FIRST, then only the values that must come from launch
        # context. Anything appearing in BOTH would be won by this dict, which
        # silently made the YAML inert for every tunable in it.
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': use_sim_time,
            'map_frame': 'map',
            'odom_frame': 'lio_init',
            'body_frame': 'base_footprint',   # /localization pose is map -> base
            'fusion_rate': 50.0,
        }],
    )

    declare_mapvox_cmd = DeclareLaunchArgument(
        'map_voxel_size', default_value='0.15',
        description='Voxel leaf the PRIOR MAP is downsampled to on load. This '
                    'is the precision floor for the whole stack: ICP cannot '
                    'localize finer than the map it matches against. Upstream '
                    'shipped 0.4 m, and measured map->odom corrections sat at '
                    '150-370 mm -- the same scale. Smaller = finer but more '
                    'points and slower ICP.')

    declare_corr_cmd = DeclareLaunchArgument(
        'max_corr_dist', default_value='0.25',
        description='ICP correspondence radius at the fine scale, and the radius '
                    'the inlier-ratio fitness is measured over -- deliberately the '
                    'same number, so fitness means "fraction that actually aligned" '
                    'rather than "fraction within a metre of anything". Upstream '
                    'used 1.0 m, which on large flat walls lets the solution slide '
                    'along the surface while every point keeps a partner: measured '
                    '423 mm of zero-mean jitter at fitness 0.98. NOTE this couples '
                    'to localization_th -- tightening the radius lowers fitness, so '
                    'the two must be tuned together.')

    declare_fovfar_cmd = DeclareLaunchArgument(
        'fov_far', default_value='10.0',
        description='Radius (m) the prior map is cropped to around the current '
                    'estimate each cycle -- the SEARCH AREA. ICP only ever sees '
                    'inside this ball, so a seed further off than this can never '
                    'recover. Bigger = more tolerant of a bad initial pose, but '
                    'more map points per ICP and more chance of latching onto a '
                    'similar-looking region elsewhere. The L2 itself only returns '
                    'to ~30 m, so beyond that you are adding map with no scan to '
                    'match it against.')

    declare_params_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(localization_share, 'config', 'localization.yaml'),
        description='YAML of runtime tunables. Point this at the SOURCE copy '
                    '(src/lio_localization/config/localization.yaml) to edit and '
                    'relaunch without rebuilding. Explicit launch arguments still '
                    'override whatever is in the file.')

    ld = LaunchDescription()
    ld.add_action(declare_map_pcd_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_localization_th_cmd)
    ld.add_action(declare_freq_cmd)
    ld.add_action(declare_mapvox_cmd)
    ld.add_action(declare_corr_cmd)
    ld.add_action(declare_fovfar_cmd)
    ld.add_action(declare_params_cmd)
    ld.add_action(OpaqueFunction(function=_check_map_pcd_exists))
    ld.add_action(sensor_tf_launch)
    ld.add_action(point_lio_launch)
    ld.add_action(global_localization)
    ld.add_action(transform_fusion)
    return ld
