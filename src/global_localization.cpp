// C++/PCL port of FAST_LIO_LOCALIZATION's global_localization.py.
//
// Upstream: HViktorTsoi/FAST_LIO_LOCALIZATION (ROS 1, python2, Open3D ICP).
// Rewritten in C++ with PCL so there is NO Open3D dependency (PCL is already
// built/linked across this workspace, and installs cleanly on Jetson). Logic is
// faithful to upstream; adaptations are only names/frames for this workspace:
//
//   * Prior map is in 'map' (leveled, floor-referenced). The LIO publishes
//     /cloud_registered in 'odom_lidar' -- its own gravity-tilted world frame,
//     named by the backend's publish.map_frame param, which is a LOAM-era
//     misnomer for "the frame I stamp my output in" and NOT a REP-105 map
//     frame. With publish_tf:=false the backend broadcasts no TF of its own, so
//     ICP of that scan against the prior map directly yields the
//     map->odom_lidar correction, published on /map_to_odom
//     (nav_msgs/Odometry, frame_id 'map'). transform_fusion turns it into the
//     map->odom_lidar TF at 50 Hz.
//   * Prior map loaded straight from a .pcd path (map_pcd param).
//   * FITNESS: Open3D's ICP fitness is the INLIER RATIO (fraction of source
//     points with a correspondence within max_corr_dist; higher=better). PCL's
//     getFitnessScore() is a mean-squared distance instead, so we compute the
//     inlier ratio ourselves (KdTree) to preserve upstream's threshold meaning.
//
// Replaces the runtime pgo_map_odom_bridge for localization runs.

#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>
#include <fstream>
#include <vector>
#include <std_srvs/srv/trigger.hpp>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;
// Point-to-plane needs a normal on the TARGET, which PointXYZ cannot carry.
using PointN = pcl::PointNormal;
using CloudN = pcl::PointCloud<PointN>;

namespace {

Eigen::Matrix4f poseToMat(const geometry_msgs::msg::Pose &p)
{
  Eigen::Quaternionf q(static_cast<float>(p.orientation.w),
                       static_cast<float>(p.orientation.x),
                       static_cast<float>(p.orientation.y),
                       static_cast<float>(p.orientation.z));
  q.normalize();
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = q.toRotationMatrix();
  m(0, 3) = static_cast<float>(p.position.x);
  m(1, 3) = static_cast<float>(p.position.y);
  m(2, 3) = static_cast<float>(p.position.z);
  return m;
}

geometry_msgs::msg::Pose matToPose(const Eigen::Matrix4f &m)
{
  geometry_msgs::msg::Pose p;
  Eigen::Matrix3f R = m.block<3, 3>(0, 0);
  Eigen::Quaternionf q(R);
  q.normalize();
  p.position.x = m(0, 3);
  p.position.y = m(1, 3);
  p.position.z = m(2, 3);
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

void voxel(const Cloud::ConstPtr &in, Cloud::Ptr &out, double leaf)
{
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.filter(*out);
}

}  // namespace

class GlobalLocalization : public rclcpp::Node
{
public:
  GlobalLocalization() : Node("lio_localization")
  {
    map_pcd_ = declare_parameter<std::string>("map_pcd", "");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_footprint");
    // /initialpose carries the ROBOT's pose in map (that is what RViz's
    // "2D Pose Estimate" publishes). The ICP below aligns /cloud_registered,
    // which is already in the ODOM frame, so the seed it needs is map -> odom.
    // Convert with (odom -> base)^-1 instead of assuming the two are the same.
    //
    // They used to BE the same: the saved map was written in the raw PGO map
    // frame, which shares odom's orientation, so an identity /initialpose was
    // accidentally a correct map -> odom seed. Now that map_batch.pcd is saved
    // gravity-leveled, they differ by the ~90 deg sensor mount, and seeding
    // without this conversion starts ICP badly misaligned -- it then converges
    // to a flipped local minimum WITH HIGH FITNESS, which looks like success.
    initial_pose_is_base_ = declare_parameter<bool>("initial_pose_is_base", true);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    MAP_VOXEL_ = declare_parameter<double>("map_voxel_size", 0.4);
    SCAN_VOXEL_ = declare_parameter<double>("scan_voxel_size", 0.1);
    FREQ_ = declare_parameter<double>("freq_localization", 0.5);
    LOCALIZATION_TH_ = declare_parameter<double>("localization_th", 0.90);
    MAX_CORR_DIST_ = declare_parameter<double>("max_corr_dist", 0.25);
    ICP_ITERS_ = declare_parameter<int>("icp_iterations", 40);
    // Motion gate (AMCL's update_min_d / update_min_a). Corrections are skipped
    // until the robot has moved this far in the odom frame since the last one.
    // 0.0 for either disables the gate on that axis; 0.0 for both restores
    // purely time-based correction.
    // Point-to-plane ICP instead of point-to-point. OFF by default: the
    // 153 mm-scatter baseline everything else is tuned against was measured
    // with point-to-point, so this must be A/B'd before it becomes the default.
    // Innovation gate: reject corrections larger than measured drift can
    // explain over the distance travelled. See the long note at the accept
    // site. innovation_gate:false disables it entirely.
    INNOV_ENABLED_ = declare_parameter<bool>("innovation_gate", true);
    // MEASURED: FAST-LIO closed a full ~165 m run to 0.1% of path length.
    DRIFT_RATE_ = declare_parameter<double>("odom_drift_rate", 0.001);
    // Multiplier on the drift budget. 10x is deliberately generous: the 0.1%
    // figure is END-POINT loop closure, which lets systematic bias cancel, so
    // true mid-run drift is larger than it implies.
    INNOV_SAFETY_ = declare_parameter<double>("innovation_safety_factor", 10.0);
    // Floor, in metres. Without it a correction taken at a standstill would get
    // a zero budget and every match would be rejected. Sized above the ICP
    // scatter a good lock produces, not above a real relocalization.
    INNOV_MIN_ = declare_parameter<double>("innovation_min", 0.30);
    // Consecutive violations before the gate yields. Too low and a corridor
    // slide is waved through; too high and real relocalization is slow.
    INNOV_MAX_REJECTS_ = declare_parameter<int>("innovation_max_rejects", 3);
    POINT_TO_PLANE_ = declare_parameter<bool>("point_to_plane", false);
    NORMAL_K_ = declare_parameter<int>("normal_k", 10);
    UPDATE_MIN_D_ = declare_parameter<double>("update_min_d", 0.25);
    UPDATE_MIN_A_ = declare_parameter<double>("update_min_a", 0.2);
    FOV_ = declare_parameter<double>("fov", 6.28);        // >pi -> ring lidar
    FOV_FAR_ = declare_parameter<double>("fov_far", 30.0);

    // Pose-uncertainty envelope reported on /map_to_odom (see poseSigmas).
    SIGMA_XY_MIN_ = declare_parameter<double>("pose_sigma_xy_min", 0.03);
    SIGMA_XY_MAX_ = declare_parameter<double>("pose_sigma_xy_max", 0.30);
    SIGMA_YAW_MIN_ = declare_parameter<double>("pose_sigma_yaw_min", 0.01);
    SIGMA_YAW_MAX_ = declare_parameter<double>("pose_sigma_yaw_max", 0.10);

    // --- seedless (global) localization ---
    // Empty disables it: the node then behaves exactly as before, requiring a
    // manual /initialpose. Point it at the optimized_poses.txt that
    // fastlio_lc_pgo writes next to map_batch.pcd.
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/cloud_registered");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom_lio");

    kf_poses_path_ = declare_parameter<std::string>("keyframe_poses", "");
    GLOBAL_SPACING_ = declare_parameter<double>("global_candidate_spacing", 3.0);
    GLOBAL_YAW_BINS_ = declare_parameter<int>("global_yaw_bins", 12);
    GLOBAL_TOP_K_ = declare_parameter<int>("global_top_k", 5);
    GLOBAL_COARSE_ITERS_ = declare_parameter<int>("global_coarse_iterations", 6);
    GLOBAL_COARSE_SCALE_ = declare_parameter<double>("global_coarse_scale", 5.0);
    // Off by default: a search that locks onto the wrong place silently is
    // worse than one that waits to be asked. Turn on for unattended bringup,
    // once you trust the acceptance gate on your map.
    AUTO_INIT_ = declare_parameter<bool>("auto_initialize", false);

    global_map_.reset(new Cloud);
    if (map_pcd_.empty()) {
      throw std::runtime_error("param 'map_pcd' (prior map .pcd path) is required");
    }
    RCLCPP_INFO(get_logger(), "Loading prior map: %s", map_pcd_.c_str());
    Cloud::Ptr raw(new Cloud);
    if (pcl::io::loadPCDFile<PointT>(map_pcd_, *raw) < 0) {
      throw std::runtime_error("failed to load map_pcd: " + map_pcd_);
    }
    voxel(raw, global_map_, MAP_VOXEL_);
    RCLCPP_INFO(get_logger(),
                "Prior map: %zu pts (voxel %.2f). Set an initial pose in RViz "
                "(2D Pose Estimate -> /initialpose).",
                global_map_->size(), MAP_VOXEL_);

    pub_map_to_odom_ = create_publisher<nav_msgs::msg::Odometry>("/map_to_odom", 1);
    pub_submap_ = create_publisher<sensor_msgs::msg::PointCloud2>("/submap", 1);
    pub_scan_in_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cur_scan_in_map", 1);

    auto qos = rclcpp::QoS(1);
    // REGRESSION FIX 2026-08-12: the odometry topic was hardcoded "/Odometry".
    // Every mapping launch in this workspace now remaps FAST-LIO's odometry
    // onto /odom_lio (the shared name across FAST-LIO, Point-LIO, FAST-LIVO2
    // and RTAB-Map), so "/Odometry" ended up with ZERO publishers and two
    // subscribers. cbOdom never fired, cur_odom_ stayed null, and
    // globalLocalization()'s first guard returned false -- which made even a
    // CORRECT manual /initialpose report "Initial registration failed", with
    // nothing in the log pointing at the real cause. Parameterised so the next
    // rename is a config change, not a silent breakage.
    sub_scan_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, qos,
      std::bind(&GlobalLocalization::cbScan, this, std::placeholders::_1));
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&GlobalLocalization::cbOdom, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribing: cloud '%s', odom '%s'.",
                cloud_topic_.c_str(), odom_topic_.c_str());
    sub_init_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 1,
      std::bind(&GlobalLocalization::cbInitialPose, this, std::placeholders::_1));

    // Candidate places for a seedless search: where the robot has actually
    // been, straight out of the mapping run.
    if (!kf_poses_path_.empty()) {
      kf_poses_ = loadKeyframePoses(kf_poses_path_);
      kf_cands_ = thinCandidates(kf_poses_, GLOBAL_SPACING_);
      if (kf_poses_.empty()) {
        RCLCPP_WARN(get_logger(),
          "keyframe_poses '%s' could not be read; global localization and "
          "/relocalize will be unavailable and an /initialpose will be required.",
          kf_poses_path_.c_str());
      } else {
        RCLCPP_INFO(get_logger(),
          "Global localization armed: %zu keyframe poses -> %zu candidates at "
          ">=%.1f m spacing, x %d yaw bins = %zu hypotheses.",
          kf_poses_.size(), kf_cands_.size(), GLOBAL_SPACING_,
          GLOBAL_YAW_BINS_, kf_cands_.size() * size_t(GLOBAL_YAW_BINS_));
      }
    }

    // Also the kidnapped-robot recovery: initialized_ was set once in
    // cbInitialPose and never reset, so there was no way to ask for a new fix.
    srv_relocalize_ = create_service<std_srvs::srv::Trigger>(
      "/relocalize",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        if (kf_poses_.empty()) {
          res->success = false;
          res->message = "No keyframe_poses loaded; set the parameter to the "
                         "optimized_poses.txt beside map_batch.pcd.";
          return;
        }
        initialized_ = false;   // stop steady-state tracking off a stale fix
        // Drop the innovation gate's baseline too. /relocalize means "I do not
        // trust where I think I am", so the last accepted pose is exactly what
        // must NOT be used to judge the plausibility of the new one -- keeping
        // it would make the gate reject the correct answer for disagreeing
        // with the fix we just disowned.
        have_last_accept_ = false;
        consec_rejects_ = 0;
        startGlobalSearch();
        res->success = true;
        res->message = "Global search started in the background; watch the log.";
      });

    T_map_to_odom_ = Eigen::Matrix4f::Identity();
    running_ = true;
    loc_thread_ = std::thread(&GlobalLocalization::locLoop, this);

    if (AUTO_INIT_ && !kf_poses_.empty()) {
      // Deferred: a scan and an odom sample must arrive first, and the search
      // itself needs the node spinning to log progress.
      auto_init_timer_ = create_wall_timer(std::chrono::seconds(2), [this]() {
        if (initialized_ || search_running_) return;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          if (!cur_scan_ || !cur_odom_) return;   // wait for data
        }
        auto_init_timer_->cancel();
        RCLCPP_INFO(get_logger(), "auto_initialize: starting global search.");
        startGlobalSearch();
      });
    }
  }

  ~GlobalLocalization() override
  {
    running_ = false;
    if (loc_thread_.joinable()) loc_thread_.join();
    if (search_thread_.joinable()) search_thread_.join();
  }

private:
  void cbOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cur_odom_ = msg;
  }

  void cbScan(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    Cloud::Ptr c(new Cloud);
    pcl::fromROSMsg(*msg, *c);  // /cloud_registered is already in the odom frame
    std::lock_guard<std::mutex> lk(mtx_);
    cur_scan_ = c;
  }

  void cbInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    Eigen::Matrix4f init = poseToMat(msg->pose.pose);
    if (initial_pose_is_base_) {
      try {
        auto tf = tf_buffer_->lookupTransform(
            odom_frame_, base_frame_, tf2::TimePointZero,
            tf2::durationFromSec(1.0));
        Eigen::Matrix4f m_odom_base = Eigen::Matrix4f::Identity();
        Eigen::Quaternionf q(tf.transform.rotation.w, tf.transform.rotation.x,
                             tf.transform.rotation.y, tf.transform.rotation.z);
        m_odom_base.block<3,3>(0,0) = q.toRotationMatrix();
        m_odom_base(0,3) = tf.transform.translation.x;
        m_odom_base(1,3) = tf.transform.translation.y;
        m_odom_base(2,3) = tf.transform.translation.z;
        init = init * m_odom_base.inverse();   // map->base * (odom->base)^-1
        RCLCPP_INFO(get_logger(),
            "Initial pose treated as %s -> %s; converted to a %s -> %s seed.",
            map_frame_.c_str(), base_frame_.c_str(),
            map_frame_.c_str(), odom_frame_.c_str());
      } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(get_logger(),
            "No %s -> %s yet (%s); using the initial pose directly as a "
            "%s -> %s seed. That is only correct if the prior map was saved in "
            "the odom frame.",
            odom_frame_.c_str(), base_frame_.c_str(), ex.what(),
            map_frame_.c_str(), odom_frame_.c_str());
      }
    }
    RCLCPP_INFO(get_logger(), "Initial pose received -> global registration...");

    // Publish the seed itself as the first map -> odom correction, BEFORE
    // attempting registration. transform_fusion withholds the transform until
    // it sees one, so without this there is no map frame until ICP accepts --
    // and RViz, whose fixed frame is map, renders nothing and cannot offer the
    // 2D Pose Estimate tool. Seeding would then require the command line.
    //
    // Unlike the identity default this replaced, the seed is a real estimate:
    // it carries the sensor mount's rotation via the odom -> base lookup above,
    // and the operator asserted the position. It is refined the moment ICP
    // accepts a match.
    publishMapToOdom(init, msg->header.stamp, -1.0);   // seed: no ICP support yet

    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!cur_scan_) {
        RCLCPP_WARN(get_logger(),
                    "No scan yet; move the robot / wait for /cloud_registered.");
        return;
      }
    }
    if (globalLocalization(init)) {
      initialized_ = true;
      RCLCPP_INFO(get_logger(), "Initialized successfully.");
    } else {
      RCLCPP_WARN(get_logger(), "Initial registration failed; try the pose again.");
    }
  }

  // Inlier ratio == Open3D ICP fitness (fraction of aligned source points with a
  // target neighbour within max_dist). Preserves upstream's threshold meaning.
  double inlierFitness(const Cloud &aligned, const Cloud::Ptr &target, double max_dist)
  {
    if (aligned.empty() || target->empty()) return 0.0;
    pcl::KdTreeFLANN<PointT> kd;
    kd.setInputCloud(target);
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    const double r2 = max_dist * max_dist;
    size_t inliers = 0;
    for (const auto &p : aligned.points) {
      if (kd.nearestKSearch(p, 1, idx, d2) > 0 && d2[0] <= r2) ++inliers;
    }
    return static_cast<double>(inliers) / static_cast<double>(aligned.size());
  }

  // Point-to-plane ICP. Minimises distance from each scan point to the PLANE
  // through its map neighbour, instead of to the neighbour itself.
  //
  // WHY IT SHOULD WIN HERE: point-to-point forces a one-to-one pairing, so on a
  // large flat wall it pulls the solution toward whichever arbitrary map point
  // happened to be nearest -- the "spurious correspondences on flat walls"
  // effect already documented against map_voxel_size, and the mechanism behind
  // the ~423 mm of measured zero-mean jitter. Point-to-plane leaves the scan
  // free to SLIDE along a surface at no cost and only penalises motion INTO or
  // OUT OF it, which is the correct constraint from a plane observation and
  // exactly the geometry of an indoor environment.
  //
  // COST: normals must be estimated on the cropped map every call. That is the
  // dominant added expense, so normal_k is kept small; the crop is already
  // bounded by fov_far.
  Eigen::Matrix4f registrationPointToPlane(const Cloud::Ptr &scan_ds,
                                           const Cloud::Ptr &map_ds,
                                           const Eigen::Matrix4f &guess,
                                           double scale, double &fitness_out,
                                           int iters)
  {
    // Too few points to fit planes through: fall back rather than return a
    // garbage transform. A sparse crop is exactly where normals are worst.
    if (map_ds->size() < 10 || scan_ds->size() < 10) {
      pcl::IterativeClosestPoint<PointT, PointT> icp;
      icp.setInputSource(scan_ds);
      icp.setInputTarget(map_ds);
      icp.setMaxCorrespondenceDistance(MAX_CORR_DIST_ * scale);
      icp.setMaximumIterations(iters > 0 ? iters : ICP_ITERS_);
      Cloud aligned;
      icp.align(aligned, guess);
      fitness_out = inlierFitness(aligned, map_ds, MAX_CORR_DIST_ * scale);
      return icp.getFinalTransformation();
    }

    // --- normals on the target -------------------------------------------
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(map_ds);
    ne.setSearchMethod(pcl::search::KdTree<PointT>::Ptr(
        new pcl::search::KdTree<PointT>()));
    ne.setKSearch(NORMAL_K_);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    CloudN::Ptr tgt(new CloudN);
    tgt->reserve(map_ds->size());
    for (size_t i = 0; i < map_ds->size(); ++i) {
      // Drop points whose normal did not resolve. Feeding a NaN normal into
      // the LLS solver poisons the whole least-squares system, so a handful of
      // bad points would corrupt an otherwise good match.
      if (!std::isfinite(normals->points[i].normal_x) ||
          !std::isfinite(normals->points[i].normal_y) ||
          !std::isfinite(normals->points[i].normal_z)) {
        continue;
      }
      PointN p;
      p.x = map_ds->points[i].x;
      p.y = map_ds->points[i].y;
      p.z = map_ds->points[i].z;
      p.normal_x = normals->points[i].normal_x;
      p.normal_y = normals->points[i].normal_y;
      p.normal_z = normals->points[i].normal_z;
      tgt->push_back(p);
    }
    if (tgt->size() < 10) {   // nearly everything failed: fall back
      pcl::IterativeClosestPoint<PointT, PointT> icp;
      icp.setInputSource(scan_ds);
      icp.setInputTarget(map_ds);
      icp.setMaxCorrespondenceDistance(MAX_CORR_DIST_ * scale);
      icp.setMaximumIterations(iters > 0 ? iters : ICP_ITERS_);
      Cloud aligned;
      icp.align(aligned, guess);
      fitness_out = inlierFitness(aligned, map_ds, MAX_CORR_DIST_ * scale);
      return icp.getFinalTransformation();
    }

    // Source carries no normals: PointToPlaneLLS reads the TARGET's, and the
    // scan is far too sparse for its own normals to be trustworthy anyway.
    CloudN::Ptr src(new CloudN);
    src->reserve(scan_ds->size());
    for (const auto &q : scan_ds->points) {
      PointN p;
      p.x = q.x; p.y = q.y; p.z = q.z;
      p.normal_x = p.normal_y = p.normal_z = 0.0f;
      src->push_back(p);
    }

    // Plain IterativeClosestPoint with an explicit point-to-plane estimator,
    // NOT IterativeClosestPointWithNormals: the latter also rejects
    // correspondences on normal agreement, which would compare against the
    // zeroed source normals above and throw away good pairs.
    pcl::IterativeClosestPoint<PointN, PointN> icp;
    icp.setTransformationEstimation(
        pcl::registration::TransformationEstimationPointToPlaneLLS<
            PointN, PointN>::Ptr(
            new pcl::registration::TransformationEstimationPointToPlaneLLS<
                PointN, PointN>()));
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.setMaxCorrespondenceDistance(MAX_CORR_DIST_ * scale);
    icp.setMaximumIterations(iters > 0 ? iters : ICP_ITERS_);

    CloudN aligned_n;
    icp.align(aligned_n, guess);
    const Eigen::Matrix4f T = icp.getFinalTransformation();

    // Fitness stays the SAME inlier-ratio metric as point-to-point, measured
    // over the same radius. It has to: localization_th is calibrated against
    // it, and changing the metric under the gate would silently re-tune the
    // acceptance threshold.
    Cloud aligned;
    aligned.reserve(aligned_n.size());
    for (const auto &q : aligned_n.points) aligned.push_back(PointT(q.x, q.y, q.z));
    fitness_out = inlierFitness(aligned, map_ds, MAX_CORR_DIST_ * scale);
    return T;
  }

  // Coarse->fine multi-scale ICP (upstream registration_at_scale).
  Eigen::Matrix4f registrationAtScale(const Cloud::Ptr &scan, const Cloud::Ptr &map,
                                      const Eigen::Matrix4f &guess, double scale,
                                      double &fitness_out, int iters = -1)
  {
    Cloud::Ptr scan_ds(new Cloud), map_ds(new Cloud);
    voxel(scan, scan_ds, SCAN_VOXEL_ * scale);
    voxel(map, map_ds, MAP_VOXEL_ * scale);

    if (POINT_TO_PLANE_) {
      return registrationPointToPlane(scan_ds, map_ds, guess, scale,
                                      fitness_out, iters);
    }

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(scan_ds);
    icp.setInputTarget(map_ds);
    // Upstream hardcoded 1.0 m at the fine scale, which is very loose: a scan
    // point pairs with a map point up to a metre away. On the large flat walls
    // this rig sees, that is exactly what lets the solution SLIDE along a
    // surface while every point still finds a partner -- measured as ~423 mm
    // of zero-mean jitter in map->odom (net/sum ratio 0.09, i.e. 91% of the
    // motion cancelled out) at fitness 0.98. Tightening it penalises sliding.
    icp.setMaxCorrespondenceDistance(MAX_CORR_DIST_ * scale);
    // iters < 0 means "use the configured count". The global search overrides it
    // with a much smaller number: that pass only RANKS hypotheses, and running
    // 40 iterations on each of several hundred is what would make a seedless
    // search take minutes instead of seconds.
    icp.setMaximumIterations(iters > 0 ? iters : ICP_ITERS_);

    Cloud aligned;
    icp.align(aligned, guess);
    Eigen::Matrix4f T = icp.getFinalTransformation();
    // Same radius as the solver, deliberately: fitness then means "fraction of
    // points that actually aligned", not "fraction within a metre of anything".
    fitness_out = inlierFitness(aligned, map_ds, MAX_CORR_DIST_ * scale);
    return T;
  }

  // Crop the prior map to the sensor FOV around the current estimate. Returns
  // points in the MAP frame (target for ICP). FOV > pi => 360 ring (distance).
  Cloud::Ptr cropMapInFov(const Eigen::Matrix4f &pose_est,
                          const geometry_msgs::msg::Pose &odom_pose)
  {
    Eigen::Matrix4f T_map_base = pose_est * poseToMat(odom_pose);
    Eigen::Matrix4f T_base_map = T_map_base.inverse();

    Cloud::Ptr in_base(new Cloud);
    pcl::transformPointCloud(*global_map_, *in_base, T_base_map);

    Cloud::Ptr fov(new Cloud);
    fov->reserve(in_base->size());
    const double far2 = FOV_FAR_ * FOV_FAR_;
    const bool ring = FOV_ > 3.14;
    for (size_t i = 0; i < in_base->size(); ++i) {
      const auto &p = in_base->points[i];
      bool keep;
      if (ring) {
        keep = (p.x * p.x + p.y * p.y) < far2;
      } else {
        double ang = std::abs(std::atan2(p.y, p.x));
        keep = (p.x > 0.0) && (p.x < FOV_FAR_) && (ang < FOV_ / 2.0);
      }
      if (keep) fov->push_back(global_map_->points[i]);  // keep in MAP frame
    }
    return fov;
  }

  void publishCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub,
                    const Cloud::Ptr &cloud, const rclcpp::Time &stamp)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = stamp;
    pub->publish(msg);
  }

  // Turn an ICP fitness into a pose covariance. fitness is the inlier ratio,
  // and it is only ever published above LOCALIZATION_TH_, so the useful range
  // is [th, 1]: rescale that to [0, 1] and interpolate between the configured
  // sigma bounds. A barely-accepted match reports the loose bound, a near
  // perfect one the tight bound. fitness < 0 means "no ICP behind this" -- the
  // /initialpose seed -- which gets the loose bound outright.
  //
  // This is a monotone confidence proxy, not a propagated covariance: the
  // inlier ratio says how much of the scan agrees with the map, not how that
  // disagreement distributes over x/y/yaw. Treat the numbers as ordering
  // information for a consumer, not as metrology.
  void poseSigmas(double fitness, double &sigma_xy, double &sigma_yaw) const
  {
    double q = 0.0;
    if (fitness >= 0.0) {
      const double span = std::max(1e-6, 1.0 - LOCALIZATION_TH_);
      q = std::clamp((fitness - LOCALIZATION_TH_) / span, 0.0, 1.0);
    }
    sigma_xy = SIGMA_XY_MAX_ + (SIGMA_XY_MIN_ - SIGMA_XY_MAX_) * q;
    sigma_yaw = SIGMA_YAW_MAX_ + (SIGMA_YAW_MIN_ - SIGMA_YAW_MAX_) * q;
  }

  // fitness < 0 => seed pose with no ICP support (see poseSigmas).
  void publishMapToOdom(const Eigen::Matrix4f &T, const rclcpp::Time &stamp,
                        double fitness)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;
    msg.child_frame_id = odom_frame_;
    msg.pose.pose = matToPose(T);

    // Row-major 6x6 over [x y z roll pitch yaw]; diagonal only. Leaving this
    // at zero (as upstream does) claims a perfect correction, which is what
    // transform_fusion used to forward onto the fused pose.
    double sigma_xy = 0.0, sigma_yaw = 0.0;
    poseSigmas(fitness, sigma_xy, sigma_yaw);
    msg.pose.covariance[0] = sigma_xy * sigma_xy;    // x
    msg.pose.covariance[7] = sigma_xy * sigma_xy;    // y
    msg.pose.covariance[14] = sigma_xy * sigma_xy;   // z
    msg.pose.covariance[21] = sigma_yaw * sigma_yaw; // roll
    msg.pose.covariance[28] = sigma_yaw * sigma_yaw; // pitch
    msg.pose.covariance[35] = sigma_yaw * sigma_yaw; // yaw

    pub_map_to_odom_->publish(msg);
  }

  // from_search: the caller already holds reg_mtx_ (globalSearch does), and the
  // failure log should stay quiet because trying many hypotheses and rejecting
  // most of them is the normal path there, not a problem.
  bool globalLocalization(const Eigen::Matrix4f &pose_estimation,
                          bool from_search = false)
  {
    // serialize registrations (initialpose callback vs background loop)
    std::unique_lock<std::mutex> reg_lk(reg_mtx_, std::defer_lock);
    if (!from_search) reg_lk.lock();

    Cloud::Ptr scan;
    nav_msgs::msg::Odometry odom;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!cur_scan_ || !cur_odom_) return false;
      scan.reset(new Cloud(*cur_scan_));
      odom = *cur_odom_;
    }

    const auto t0 = now();
    Cloud::Ptr fov = cropMapInFov(pose_estimation, odom.pose.pose);

    double f_coarse = 0.0, fitness = 0.0;
    Eigen::Matrix4f transf = registrationAtScale(scan, fov, pose_estimation, 5.0, f_coarse);
    transf = registrationAtScale(scan, fov, transf, 1.0, fitness);
    const double dt_ms = (now() - t0).seconds() * 1e3;

    // RViz debug clouds (both in map frame), as upstream publishes them.
    publishCloud(pub_submap_, fov, odom.header.stamp);
    Cloud::Ptr scan_in_map(new Cloud);
    pcl::transformPointCloud(*scan, *scan_in_map, transf);
    publishCloud(pub_scan_in_map_, scan_in_map, odom.header.stamp);

    if (fitness > LOCALIZATION_TH_) {
      // --- INNOVATION GATE --------------------------------------------------
      // Fitness alone cannot separate a true lock from a confident WRONG one:
      // in a corridor the ICP cost surface is nearly flat along the corridor
      // axis, so the solution slides while every point still lands on a wall
      // and the inlier ratio reads near-perfect. MEASURED: the point-to-plane
      // arm scored 0.97-1.00 while positioning WORSE than point-to-point.
      //
      // So gate on PLAUSIBILITY instead, using the one thing fitness does not
      // know: how far the robot has actually travelled. Odometry drifts at a
      // measured 0.001 (0.1% of path length, full-run loop closure), so over a
      // known distance there is a bound on how much map -> odom can legitimately
      // need to move. A correction far beyond that bound is not correcting
      // drift -- it is relocating, which for a robot that has driven 5 m in a
      // known direction means the match is wrong.
      //
      // This is the standard innovation / Mahalanobis gate of any Kalman-style
      // estimator, sized from a measured drift rate rather than a covariance.
      //
      // ESCAPE HATCH: a genuine relocalization (robot bumped, pushed, carried)
      // also violates the gate, and must not be locked out forever. Consecutive
      // violations are counted, and once innovation_max_rejects of them agree
      // that the pose has moved, the correction is accepted. One outlier is
      // noise; several in a row are the world.
      bool gated = false;
      if (INNOV_ENABLED_ && !from_search && have_last_accept_) {
        Eigen::Matrix4f prev;
        {
          std::lock_guard<std::mutex> lk(mtx_);
          prev = T_map_to_odom_;
        }
        // How far the ROBOT moved since the last accepted correction, in odom
        // (continuous by REP-105, and unaffected by our own corrections).
        const Eigen::Matrix4f odom_now = poseToMat(odom.pose.pose);
        const double d_travel =
            (last_accept_odom_.inverse() * odom_now).block<3, 1>(0, 3).norm();
        // How much map -> odom is being asked to jump.
        const double innovation =
            (transf.block<3, 1>(0, 3) - prev.block<3, 1>(0, 3)).norm();
        // Budget: drift over that distance, times a safety factor, with a floor
        // so ICP scatter at a standstill is not gated to zero.
        const double budget =
            std::max(INNOV_MIN_, DRIFT_RATE_ * d_travel * INNOV_SAFETY_);

        if (innovation > budget) {
          ++consec_rejects_;
          if (consec_rejects_ < INNOV_MAX_REJECTS_) {
            RCLCPP_WARN(get_logger(),
                "REJECTED implausible correction: %.2f m jump after %.2f m "
                "travelled (budget %.2f m), fitness %.3f. %d/%d consecutive -- "
                "accepting anyway at %d.",
                innovation, d_travel, budget, fitness,
                consec_rejects_, INNOV_MAX_REJECTS_, INNOV_MAX_REJECTS_);
            return false;
          }
          RCLCPP_WARN(get_logger(),
              "Accepting %.2f m correction after %d consecutive violations -- "
              "treating as a genuine relocalization, not an outlier.",
              innovation, consec_rejects_);
          gated = true;
        }
      }
      if (!gated) consec_rejects_ = 0;

      {
        std::lock_guard<std::mutex> lk(mtx_);
        T_map_to_odom_ = transf;
      }
      last_accept_odom_ = poseToMat(odom.pose.pose);
      have_last_accept_ = true;
      publishMapToOdom(transf, odom.header.stamp, fitness);
      RCLCPP_INFO(get_logger(), "matched (fitness %.3f, %.0f ms)", fitness, dt_ms);
      return true;
    }
    if (!from_search)
      RCLCPP_WARN(get_logger(), "no match (fitness %.3f < %.2f)", fitness, LOCALIZATION_TH_);
    return false;
  }

  // Correction loop. MOTION-GATED, not purely time-driven.
  //
  // WHY. A stationary robot generates no new information, so an ICP match
  // computed at rest is measurement noise being injected straight into
  // map -> odom: the pose visibly wanders while the robot is standing still.
  // Running on a bare timer also spends CPU re-deriving a correction that
  // cannot have changed. AMCL has gated on distance travelled since forever
  // (update_min_d 0.25 m, update_min_a 0.2 rad); this is the same rule.
  //
  // FREQ_ still bounds the loop from ABOVE -- it is the maximum correction
  // rate, not the rate itself. Motion decides whether each tick does work.
  //
  // Set update_min_d and update_min_a to 0.0 to restore the old purely
  // time-based behaviour.
  void locLoop()
  {
    const auto period = std::chrono::duration<double>(1.0 / FREQ_);
    bool have_last = false;
    Eigen::Matrix4f last_odom = Eigen::Matrix4f::Identity();

    while (running_ && rclcpp::ok()) {
      std::this_thread::sleep_for(period);
      if (!initialized_) continue;

      Eigen::Matrix4f guess, odom_now;
      bool have_odom = false;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        guess = T_map_to_odom_;
        if (cur_odom_) {
          odom_now = poseToMat(cur_odom_->pose.pose);
          have_odom = true;
        }
      }

      // Measure movement in the ODOM frame, which is continuous by REP-105.
      // Doing it in map would fold our own corrections back into the gate and
      // let ICP noise masquerade as robot motion, re-triggering itself.
      if (have_odom && (UPDATE_MIN_D_ > 0.0 || UPDATE_MIN_A_ > 0.0)) {
        if (have_last) {
          const Eigen::Matrix4f delta = last_odom.inverse() * odom_now;
          const double d = delta.block<3, 1>(0, 3).norm();
          // Rotation angle of the delta, from the trace of its rotation block.
          const double tr = delta.block<3, 3>(0, 0).trace();
          const double a = std::acos(
              std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0)));
          if (d < UPDATE_MIN_D_ && a < UPDATE_MIN_A_) {
            // Below threshold: hold the last correction. Holding is correct,
            // not lazy -- map -> odom is only stale to the extent odometry has
            // drifted, and by definition it has barely moved.
            continue;
          }
        }
        last_odom = odom_now;
        have_last = true;
      }

      globalLocalization(guess);  // scan already in odom frame
    }
  }

  // ---- global (seedless) localization -----------------------------------
  //
  // The rest of this node is LOCAL: registrationAtScale's capture radius is
  // max_corr_dist * scale, and cropMapInFov crops the prior to fov_far around
  // the guess, so a seed more than ~1 m and a few degrees off can never
  // recover. Without the block below the node needs a human to publish
  // /initialpose before it will ever produce a map frame, and initialized_ is
  // never reset, so there is no kidnapped-robot recovery either.
  //
  // Approach: the mapping run already writes optimized_poses.txt (KITTI 3x4
  // per line) next to map_batch.pcd. Those keyframe poses are, by construction,
  // places the robot has actually been -- a far better candidate set than a
  // blind grid over free space. Sweep them x yaw bins, score each cheaply,
  // then verify the best few with the same two-stage registration and the same
  // acceptance gate used in steady state. No new dependency, no new file
  // format, and it reuses machinery that is already tuned.
  //
  // Scan Context would let us propose ~10 candidates instead of sweeping, but
  // its descriptor DB is never saved by the PGO node (saveScancontextAndKeys is
  // called nowhere), and its known weakness is self-similar geometry -- which a
  // repetitive indoor office is. It belongs here later as an ACCELERATOR, not
  // as the decider.
  std::vector<Eigen::Matrix4f> loadKeyframePoses(const std::string &path)
  {
    std::vector<Eigen::Matrix4f> out;
    std::ifstream f(path);
    if (!f.is_open()) return out;
    double v[12];
    while (f >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]
             >> v[6] >> v[7] >> v[8] >> v[9] >> v[10] >> v[11]) {
      Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) m(r, c) = float(v[r * 4 + c]);
      out.push_back(m);
    }
    return out;
  }

  // Thin the candidate list so entries are at least `spacing` apart. A 165 m
  // indoor run yields ~1100 keyframes; at 12 yaw bins that would be >13000
  // registrations. Keyframes also cluster wherever the robot paused or turned,
  // which adds candidates without adding coverage.
  std::vector<Eigen::Matrix4f> thinCandidates(const std::vector<Eigen::Matrix4f> &in,
                                              double spacing) const
  {
    std::vector<Eigen::Matrix4f> out;
    for (const auto &m : in) {
      bool far = true;
      for (const auto &k : out) {
        if ((m.block<3, 1>(0, 3) - k.block<3, 1>(0, 3)).norm() < spacing) { far = false; break; }
      }
      if (far) out.push_back(m);
    }
    return out;
  }

  // Returns map->odom on success. Runs on its own thread; can take seconds.
  bool globalSearch()
  {
    std::lock_guard<std::mutex> reg_lk(reg_mtx_);

    if (kf_poses_.empty()) {
      RCLCPP_ERROR(get_logger(),
        "Global localization needs keyframe poses, but '%s' held none. It is "
        "written by fastlio_lc_pgo next to map_batch.pcd; set keyframe_poses "
        "to that file, or send /initialpose manually.", kf_poses_path_.c_str());
      return false;
    }

    Cloud::Ptr scan;
    nav_msgs::msg::Odometry odom;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!cur_scan_ || !cur_odom_) {
        RCLCPP_WARN(get_logger(), "No scan/odom yet; cannot search.");
        return false;
      }
      scan.reset(new Cloud(*cur_scan_));
      odom = *cur_odom_;
    }

    // The scan arrives in the odom frame. A candidate keyframe pose is where
    // the BASE was in map, so the map->odom guess that would put the robot
    // there is  T_map_odom = T_map_base * T_odom_base^-1.
    const Eigen::Matrix4f T_odom_base = poseToMat(odom.pose.pose);
    const Eigen::Matrix4f T_odom_base_inv = T_odom_base.inverse();

    Cloud::Ptr scan_coarse(new Cloud);
    voxel(scan, scan_coarse, std::max(SCAN_VOXEL_ * 4.0, 0.4));

    const auto t0 = now();
    struct Cand { double score; Eigen::Matrix4f T; };
    std::vector<Cand> scored;
    const int bins = std::max(1, GLOBAL_YAW_BINS_);
    int tried = 0;

    for (const auto &kf : kf_cands_) {
      for (int b = 0; b < bins; ++b) {
        const double yaw = 2.0 * M_PI * double(b) / double(bins);
        Eigen::Matrix4f R = Eigen::Matrix4f::Identity();
        R(0, 0) = float(std::cos(yaw)); R(0, 1) = float(-std::sin(yaw));
        R(1, 0) = float(std::sin(yaw)); R(1, 1) = float(std::cos(yaw));
        // rotate about the candidate's own origin
        Eigen::Matrix4f T_map_base = kf;
        T_map_base.block<3, 3>(0, 0) = (R.block<3, 3>(0, 0) * kf.block<3, 3>(0, 0)).eval();
        const Eigen::Matrix4f guess = T_map_base * T_odom_base_inv;

        Cloud::Ptr fov = cropMapInFov(guess, odom.pose.pose);
        if (fov->size() < 100) continue;
        double f = 0.0;
        // Cheap pass: heavily downsampled scan, few iterations. Only ranking.
        registrationAtScale(scan_coarse, fov, guess, GLOBAL_COARSE_SCALE_, f,
                            GLOBAL_COARSE_ITERS_);
        scored.push_back({f, guess});
        ++tried;
        if (!running_) return false;
      }
    }
    if (scored.empty()) {
      RCLCPP_ERROR(get_logger(), "Global search produced no candidates.");
      return false;
    }

    std::sort(scored.begin(), scored.end(),
              [](const Cand &a, const Cand &b) { return a.score > b.score; });
    const int keep = std::min<int>(GLOBAL_TOP_K_, int(scored.size()));
    RCLCPP_INFO(get_logger(),
      "Global search: %d hypotheses over %zu places x %d yaw bins in %.1f s; "
      "best coarse score %.3f, verifying top %d.",
      tried, kf_cands_.size(), bins, (now() - t0).seconds(), scored[0].score, keep);

    // Verify the survivors with the SAME two-stage registration and the SAME
    // gate used in steady state -- a confident wrong lock is worse than none.
    for (int i = 0; i < keep; ++i) {
      if (!running_) return false;
      if (globalLocalization(scored[i].T, /*from_search=*/true)) {
        RCLCPP_INFO(get_logger(),
          "Global localization SUCCEEDED on hypothesis %d/%d.", i + 1, keep);
        return true;
      }
    }
    RCLCPP_WARN(get_logger(),
      "Global localization FAILED: no hypothesis passed localization_th (%.2f). "
      "The robot may be somewhere the prior map does not cover, or the map is "
      "stale. Send /initialpose, or call /relocalize again after moving.",
      LOCALIZATION_TH_);
    return false;
  }

  void startGlobalSearch()
  {
    if (search_running_.exchange(true)) {
      RCLCPP_WARN(get_logger(), "A global search is already running.");
      return;
    }
    if (search_thread_.joinable()) search_thread_.join();
    search_thread_ = std::thread([this]() {
      const bool ok = globalSearch();
      if (ok) initialized_ = true;
      search_running_ = false;
    });
  }

  // params
  std::string map_pcd_, map_frame_, odom_frame_, base_frame_;
  std::string kf_poses_path_, cloud_topic_, odom_topic_;
  std::vector<Eigen::Matrix4f> kf_poses_, kf_cands_;
  double UPDATE_MIN_D_{0.25}, UPDATE_MIN_A_{0.2};
  bool POINT_TO_PLANE_{false};
  bool INNOV_ENABLED_{true};
  double DRIFT_RATE_{0.001}, INNOV_SAFETY_{10.0}, INNOV_MIN_{0.30};
  int INNOV_MAX_REJECTS_{3};
  Eigen::Matrix4f last_accept_odom_{Eigen::Matrix4f::Identity()};
  bool have_last_accept_{false};
  int consec_rejects_{0};
  int NORMAL_K_{10};
  int GLOBAL_YAW_BINS_{12}, GLOBAL_TOP_K_{5}, GLOBAL_COARSE_ITERS_{6};
  double GLOBAL_COARSE_SCALE_{5.0}, GLOBAL_SPACING_{3.0};
  bool AUTO_INIT_{false};
  std::atomic<bool> search_running_{false};
  std::thread search_thread_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_relocalize_;
  rclcpp::TimerBase::SharedPtr auto_init_timer_;
  bool initial_pose_is_base_{true};
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  double MAP_VOXEL_, SCAN_VOXEL_, FREQ_, LOCALIZATION_TH_, FOV_, FOV_FAR_;
  double MAX_CORR_DIST_{0.25};
  double SIGMA_XY_MIN_, SIGMA_XY_MAX_, SIGMA_YAW_MIN_, SIGMA_YAW_MAX_;
  int ICP_ITERS_{40};

  // state
  Cloud::Ptr global_map_;
  Cloud::Ptr cur_scan_;
  nav_msgs::msg::Odometry::SharedPtr cur_odom_;
  Eigen::Matrix4f T_map_to_odom_;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> running_{false};
  std::mutex mtx_;       // guards cur_scan_/cur_odom_/T_map_to_odom_
  std::mutex reg_mtx_;   // serializes globalLocalization()
  std::thread loc_thread_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_map_to_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_scan_in_map_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<GlobalLocalization>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("lio_localization"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
