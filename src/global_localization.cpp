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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

using PointT = pcl::PointXYZ;
using Cloud = pcl::PointCloud<PointT>;

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
    FOV_ = declare_parameter<double>("fov", 6.28);        // >pi -> ring lidar
    FOV_FAR_ = declare_parameter<double>("fov_far", 30.0);

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
    sub_scan_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", qos,
      std::bind(&GlobalLocalization::cbScan, this, std::placeholders::_1));
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", qos,
      std::bind(&GlobalLocalization::cbOdom, this, std::placeholders::_1));
    sub_init_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 1,
      std::bind(&GlobalLocalization::cbInitialPose, this, std::placeholders::_1));

    T_map_to_odom_ = Eigen::Matrix4f::Identity();
    running_ = true;
    loc_thread_ = std::thread(&GlobalLocalization::locLoop, this);
  }

  ~GlobalLocalization() override
  {
    running_ = false;
    if (loc_thread_.joinable()) loc_thread_.join();
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
    publishMapToOdom(init, msg->header.stamp);

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

  // Coarse->fine multi-scale ICP (upstream registration_at_scale).
  Eigen::Matrix4f registrationAtScale(const Cloud::Ptr &scan, const Cloud::Ptr &map,
                                      const Eigen::Matrix4f &guess, double scale,
                                      double &fitness_out)
  {
    Cloud::Ptr scan_ds(new Cloud), map_ds(new Cloud);
    voxel(scan, scan_ds, SCAN_VOXEL_ * scale);
    voxel(map, map_ds, MAP_VOXEL_ * scale);

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
    icp.setMaximumIterations(ICP_ITERS_);

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

  void publishMapToOdom(const Eigen::Matrix4f &T, const rclcpp::Time &stamp)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;
    msg.child_frame_id = odom_frame_;
    msg.pose.pose = matToPose(T);
    pub_map_to_odom_->publish(msg);
  }

  bool globalLocalization(const Eigen::Matrix4f &pose_estimation)
  {
    // serialize registrations (initialpose callback vs background loop)
    std::lock_guard<std::mutex> reg_lk(reg_mtx_);

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
      {
        std::lock_guard<std::mutex> lk(mtx_);
        T_map_to_odom_ = transf;
      }
      publishMapToOdom(transf, odom.header.stamp);
      RCLCPP_INFO(get_logger(), "matched (fitness %.3f, %.0f ms)", fitness, dt_ms);
      return true;
    }
    RCLCPP_WARN(get_logger(), "no match (fitness %.3f < %.2f)", fitness, LOCALIZATION_TH_);
    return false;
  }

  void locLoop()
  {
    const auto period = std::chrono::duration<double>(1.0 / FREQ_);
    while (running_ && rclcpp::ok()) {
      std::this_thread::sleep_for(period);
      if (!initialized_) continue;
      Eigen::Matrix4f guess;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        guess = T_map_to_odom_;
      }
      globalLocalization(guess);  // scan already in odom frame
    }
  }

  // params
  std::string map_pcd_, map_frame_, odom_frame_, base_frame_;
  bool initial_pose_is_base_{true};
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  double MAP_VOXEL_, SCAN_VOXEL_, FREQ_, LOCALIZATION_TH_, FOV_, FOV_FAR_;
  double MAX_CORR_DIST_{0.25};
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
