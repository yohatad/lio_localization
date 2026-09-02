# lio_localization

Prior-map localization for **FAST-LIO and Point-LIO** — a ROS 2 **C++/PCL** port
of [HViktorTsoi/FAST_LIO_LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION),
adapted to this repo's frame contract. **No Open3D** — the ICP runs on PCL
(already built/linked across this workspace and clean to install on Jetson).

A LIO backend gives you drift-y odometry; this package registers its live scan
against a **saved point-cloud map** and publishes the `map → lio_init`
correction, so Nav2 localizes in a fixed, reproducible map frame. **No PGO / no
RTAB-Map at runtime.**

```
map ─(global_localization: ICP vs prior .pcd, ~0.5 Hz)→ lio_init
        │  (published as /map_to_odom, rebroadcast as TF @50 Hz by transform_fusion)
        └─(lio_odom_bridge)→ base_footprint ─(pepper_sensor_tf, static)→ l2lidar_frame_imu
```

## Either backend

The package is backend-agnostic; only the odometry topic name differs, and the
launch file remaps it:

| launch | backend | odometry topic |
|---|---|---|
| `fastlio_localization_l2.launch.py` | FAST-LIO | `/Odometry` (what both nodes hardcode) |
| `pointlio_localization_l2.launch.py` | Point-LIO | `/aft_mapped_to_init`, remapped onto `/Odometry` |

`/cloud_registered` is published under the identical name, in the same frame, by
both — so it needs no remap. The remap is applied at launch rather than in the
source so one build serves either backend.

**The prior map is backend-agnostic too.** `map_pcd` is just environment
geometry: a map built by the FAST-LIO PGO run localizes Point-LIO perfectly
well, which is why both launches default to the same `.pcd`.

## Nodes

| Node | Does |
|------|------|
| `global_localization` | Loads the prior `.pcd`, PCL-ICP-matches `/cloud_registered` to it, publishes `/map_to_odom` (`nav_msgs/Odometry`, frame `map`). C++. |
| `transform_fusion` | Rebroadcasts the latest `map→lio_init` as **TF at 50 Hz** so lookups stay fresh between ICP updates; also republishes fused pose on `/localization/pose`, applying the `<LIO body>→base_footprint` extrinsic to the pose and twist. C++. |

## 1. Build (no runtime pip deps)

```bash
colcon build --packages-select lio_localization
```

Everything it needs — PCL, `pcl_conversions`, Eigen, tf2 — is already in the
workspace. There is **nothing to pip-install** (this is the whole reason for the
C++ rewrite; the earlier rclpy version needed Open3D, which is painful on Jetson
aarch64). It's a C++ `ament_cmake` package, so build without `--symlink-install`.

## 2. Prepare the prior map

Map once with the loop-closure pipeline and use its saved cloud:

- PGO map: `/home/yoha/Lidar/run_l2_lc/pgo_output/map_batch.pcd` (default), or
- FAST-LIO map: `/home/yoha/Lidar/run_l2_fastlio/l2_fastlio_map.pcd`.

The node voxel-downsamples it (`map_voxel_size`, default 0.4 m) on load. The
launch file checks the file exists up front, so a missing map fails with an
actionable message instead of a C++ exception.

## 3. Run

Bag replay:

```bash
ros2 launch lio_localization fastlio_localization_l2.launch.py \
    map_pcd:=/home/yoha/Lidar/run_l2_lc/pgo_output/map_batch.pcd
ros2 bag play <bag> --clock --topics /points /imu/data
```

On the real robot — `use_sim_time:=false`, and something must actually produce
the sensor data:

```bash
ros2 launch l2lidar_node l2lidar.launch.py            # /points + /imu/data
ros2 launch lio_localization fastlio_localization_l2.launch.py use_sim_time:=false
```

`use_sim_time` defaults to `true` (bag replay). Left at the default on hardware
with no `/clock` publisher, every node blocks forever on a clock that never
ticks.

### Set the initial pose

ICP needs a rough starting guess. In RViz click **2D Pose Estimate** at the
robot's true location (publishes `/initialpose`), or publish it directly:

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  '{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}'
```

**What you publish is the ROBOT's pose in `map`**, not a `map → lio_init`
seed. `initial_pose_is_base` (default true) makes `global_localization` compose
out the current `lio_init → base_footprint` itself. This matters: before that
parameter existed, `/initialpose` was implicitly a map→odom seed, which broke
once saved maps moved into the leveled frame.

The node runs a coarse→fine global registration; on success
(`fitness > localization_th`) it locks on and tracks at `freq_localization`.
Watch the log for `matched (fitness …)`.

**Verify the lock-on visually.** The default RViz config (`rviz/localization.rviz`)
is built for exactly this — Fixed Frame `map`, with:

- `/submap` (grey) — the FOV-cropped slice of the prior map being matched against.
- `/cur_scan_in_map` (rainbow) — the live scan through the current estimate.
  **Their overlap is the localization quality**; no metric needed, a bad lock is
  visible.
- `/localization/pose` — fused `map → base_footprint` pose. Genuinely
  `base_footprint`: the `<LIO body> → base_footprint` extrinsic is looked up
  from TF once and applied, and the twist is rotated into `base_footprint`
  with its lever-arm term. Withheld (with a throttled warning) until that
  lookup succeeds — the `map → lio_init` TF is unaffected. Pose covariance
  carries the ICP fitness via `pose_sigma_*` in `config/localization.yaml`; it
  does not model LIO drift between corrections.

Fixed Frame is `map`, not `odom`: the correction lives in `map → lio_init`, so
viewed from `odom` each ICP update lurches the *world* around a stationary
robot — the same information rendered backwards.

## Tuning (launch args / params)

| Param | Default | Notes |
|-------|---------|-------|
| `localization_th` | 0.90 | Min ICP **inlier-ratio** fitness to accept — fraction of scan points with a map neighbour within the correspondence distance (computed via KdTree to match Open3D's `fitness` semantics, since PCL's own `getFitnessScore()` is a distance, not a ratio). **Lower to 0.6–0.8** if the L2 scan only partly overlaps the map. |
| `map_voxel_size` / `scan_voxel_size` | 0.4 / 0.1 | Bigger = faster, coarser. |
| `freq_localization` | 0.5 Hz | ICP correction rate (bursty CPU). |
| `fov` / `fov_far` | 6.28 / 30 | `>π` = 360° ring lidar → distance-only map crop within `fov_far` m. |
| `initial_pose_is_base` | true | Treat `/initialpose` as the robot's pose in `map` (see above). |

## Frame notes

See `pepper_slam/FRAMES.md` for the full contract. What matters here:

- This package owns **`map → lio_init`**. `lio_init` is the LIO's own,
  gravity-tilted world frame; `map` is leveled and floor-referenced.
- Because `lio_init` can have only one parent and `transform_fusion` claims
  it, the leveled `odom` is published as a **child** here
  (`level_frame_as_child:=true`) rather than as `lio_init`'s parent, which is
  how the mapping stacks do it. Leveling stays **on** — the costmaps and
  collision monitor need a gravity-aligned, floor-referenced frame.
- This **replaces** `pgo_map_odom_bridge` for localization runs. Don't run both.

## Status / caveats

- PCL point-to-point ICP with a coarse→fine schedule (scale 5 → 1), matching
  upstream. For a tilted, partial-overlap L2 scan you'll likely tune
  `localization_th` down.
- Localization runs on a background `std::thread` at `freq_localization` (like
  upstream's thread); callbacks only stash the latest scan/odom under a mutex, so
  the ~0.5 Hz ICP never blocks TF/topic handling.
- No automatic relocalization on divergence yet — if it loses lock, re-set the
  2D Pose Estimate. (A fitness-triggered global search is a sensible next add.)
- Verified on bag replay (July_22) for both backends. **Not yet run on the real
  robot.**
