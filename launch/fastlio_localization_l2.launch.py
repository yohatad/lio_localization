import os

from ament_index_python.packages import (
    PackageNotFoundError, get_package_share_directory)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

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

    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value=os.path.join(nav_share, 'pcd', 'pepper_map_lc.pcd'),
        description='Prior map .pcd to localize against (the loop-closed PGO map).'
    )
    # Enables SEEDLESS localization. Without it the node can only start from a
    # manual /initialpose, because its ICP is purely local: the coarse pass
    # captures ~max_corr_dist * 5 and cropMapInFov crops the prior to fov_far
    # around the guess, so a seed more than a few metres out can never recover.
    # These are the keyframe poses PGO writes beside map_batch.pcd -- places the
    # robot has actually been, which is a far better hypothesis set than a blind
    # grid over free space. Also what /relocalize searches.
    declare_kf_poses_cmd = DeclareLaunchArgument(
        'keyframe_poses',
        default_value=os.path.join(nav_share, 'pcd', 'pepper_map_lc_poses.txt'),
        description='KITTI-format keyframe poses from the mapping run, used as '
                    'candidates for global localization and /relocalize. Empty '
                    'disables both (manual /initialpose only). Must come from '
                    'the SAME run as map_pcd.'
    )
    declare_auto_init_cmd = DeclareLaunchArgument(
        'auto_initialize', default_value='false',
        description='Run the global search automatically at startup instead of '
                    'waiting for /initialpose or /relocalize. Off by default: a '
                    'silent lock onto the wrong place is worse than waiting.'
    )
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        # A LOCALIZATION view (Fixed Frame 'map', submap vs scan overlap), not
        # FAST-LIO's mapping config -- that one is fixed to 'odom', where every
        # ICP correction moves the map around a stationary robot.
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
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config. l2_rsimu.yaml drives the estimator from '
                    'the RealSense IMU (/camera/imu, recommended and what the '
                    'prior map was built with); l2.yaml uses the L2 s own '
                    '(/imu/data). Change lidar_imu_frame to match.')
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Static frame the estimated body corresponds to. '
                    'camera_imu_optical_frame for l2_rsimu.yaml, '
                    'l2lidar_frame_imu for l2.yaml.')
    # localization_th and max_corr_dist USED TO BE DECLARED HERE. They were dead:
    # the global_localization Node's parameter dict deliberately stays minimal so
    # the YAML is not shadowed, and neither name appeared in it, so passing
    # localization_th:=0.5 on the command line did nothing while looking like it
    # worked. Both live in config/localization.yaml, which is now the only place
    # they can be set. They are also strongly coupled -- tightening max_corr_dist
    # lowers the fitness a GOOD lock scores -- so changing one without the other
    # is what made every match get rejected; keep them together in one file.

    # base_footprint -> l2lidar_frame -> l2lidar_frame_imu (+ cams). The lio bridge
    # needs the static base_footprint -> l2lidar_frame_imu to close odom->base_footprint.
    # scope must be 'all' on BAG REPLAY and 'mount' on the robot, and the two
    # answer the same question: is a RealSense driver running? Derived from
    # use_sim_time so it cannot be forgotten -- which matters now that the
    # default config is l2_rsimu.yaml, whose body frame is
    # camera_imu_optical_frame. Under the 'mount' default that frame is never
    # published on a bag (no driver), so lio_odom_bridge cannot close
    # odom -> base_footprint and the tree comes up in two halves.
    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensor_tf_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # FAST-LIO odometry ONLY (no PGO). Publishes /Odometry + /cloud_registered
    # in the 'lio_init' frame; lio_odom_bridge closes
    # lio_init -> base_footprint.
    #
    # 2026-08-22: config_file was HARDCODED to l2.yaml, i.e. the L2's own IMU,
    # while the map this localizes against was built with l2_rsimu.yaml and
    # every bag_test launch already defaulted to the RealSense. The L2 gyro
    # cancels rotation about the gravity axis below ~16 deg/s and cost 139 deg
    # of heading over a 744 s run (utils/L2_IMU/REPORT.md) -- and a turn is
    # exactly when it drops out AND when scan overlap is worst, so it is the
    # worst possible moment to be dead-reckoning between 0.5 Hz ICP fixes.
    # Now an argument, defaulting to the RealSense like everything else.
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={
            'config_file': LaunchConfiguration('config_file'),
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
        }.items())

    # Registers /cloud_registered (odom frame) against the prior map, producing map -> lio_init.
    global_localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='fast_lio_localization',
        output='screen',
        # params_file FIRST, then only the values that must come from launch
        # context. Anything appearing in BOTH would be won by this dict, which
        # silently made the YAML inert for every tunable in it.
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': use_sim_time,
            'map_pcd': map_pcd,
            'keyframe_poses': LaunchConfiguration('keyframe_poses'),
            'auto_initialize': ParameterValue(
                LaunchConfiguration('auto_initialize'), value_type=bool),
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


    # REMOVED as dead: max_corr_dist, freq_localization, map_voxel_size and
    # fov_far were all declared here but never referenced, so none of them
    # reached the node. Two of them also disagreed with the YAML that was
    # actually in force -- this file advertised freq_localization 2.0 and
    # map_voxel_size 0.15 while the node ran at 1.0 and 0.4. They live in
    # config/localization.yaml.


    declare_params_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(localization_share, 'config', 'localization.yaml'),
        description='YAML of runtime tunables. Point this at the SOURCE copy '
                    '(src/lio_localization/config/localization.yaml) to edit and '
                    'relaunch without rebuilding. Explicit launch arguments still '
                    'override whatever is in the file.')

    ld = LaunchDescription()
    ld.add_action(declare_map_pcd_cmd)
    ld.add_action(declare_kf_poses_cmd)
    ld.add_action(declare_auto_init_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_lidar_imu_frame_cmd)
    ld.add_action(declare_params_cmd)
    ld.add_action(OpaqueFunction(function=_check_map_pcd_exists))
    ld.add_action(sensor_tf_launch)
    # odom -> base_footprint. FAST_LIO's mapping.launch.py no longer starts this
    # (see FAST_LIO d8b274c): it was Pepper glue in a launch file shared with
    # every other FAST-LIO sensor config.
    lio_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('pepper_slam'),
                         'launch', 'lio_odom_bridge.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'config_file': LaunchConfiguration('config_file'),
            'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
            # 'odom' (leveled) is a CHILD of lio_init here: transform_fusion owns
            # map -> lio_init, so it cannot also be lio_init's parent. Leveling
            # stays ON -- the costmaps and collision monitor need a
            # gravity-aligned, floor-referenced frame and raw lio_init is neither.
            'bridge_level_frame': 'true',
            'level_frame_as_child': 'true',
        }.items())

    ld.add_action(fast_lio_launch)
    ld.add_action(lio_bridge_launch)
    ld.add_action(global_localization)
    ld.add_action(transform_fusion)
    return ld
