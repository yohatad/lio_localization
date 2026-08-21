// C++ port of FAST_LIO_LOCALIZATION's transform_fusion.py.
//
// Faithful to upstream (HViktorTsoi/FAST_LIO_LOCALIZATION): broadcasts the
// map->odom_lidar TF at FREQ_PUB_LOCALIZATION (50 Hz) from the latest
// /map_to_odom correction (global_localization publishes it at only ~0.5 Hz),
// so TF lookups stay fresh between ICP updates while the LIO's
// odom_lidar->body supplies the smooth high-rate motion underneath. Also
// republishes the fused pose map->body on /localization/pose, using the odom's
// own stamp.
//
// Adaptation: upstream copies the LIO twist verbatim and stamps its own
// body_frame on child_frame_id, but the LIO body is the IMU/lidar on the mast,
// not the footprint. Both the pose and the twist are therefore the LIDAR's,
// mislabelled. This port composes the <LIO body> -> body_frame extrinsic from
// TF onto the pose and rotates the twist (with lever arm) into body_frame, so
// the message describes the frame it names. Upstream also leaves the pose
// covariance at zero -- an assertion of perfection -- where this port forwards
// the fitness-scaled covariance that global_localization now stamps on
// /map_to_odom.
//
// Adaptation: upstream broadcasts map -> 'camera_init'. That is stock
// FAST-LIO's publish.map_frame default -- a misnomer inherited from LOAM: the
// parameter only sets the header.frame_id FAST-LIO stamps on /Odometry, /path
// and its clouds, i.e. the LIO's OWN world frame. FAST-LIO does no loop closure
// or relocalization, so it cannot produce a REP-105 'map' frame at all.
//
// This workspace sets publish.map_frame:="odom_lidar" (see FAST_LIO/config/
// l2.yaml), naming that frame for what it is: the LIO's native, gravity-tilted
// odometry frame. It must NOT be plain 'odom' -- that name now belongs to the
// leveled, floor-referenced frame published by lio_map_odom_bridge, and having
// the tilted output claim it is exactly the bug that saved a 90-deg-tilted map.
//
// So this node broadcasts map -> odom_lidar; both frames are parameterised
// (map_frame / odom_frame below). See pepper_slam/FRAMES.md.

#include <array>
#include <chrono>
#include <memory>
#include <mutex>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

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

Eigen::Matrix4f transformToMat(const geometry_msgs::msg::Transform &t)
{
  Eigen::Quaternionf q(static_cast<float>(t.rotation.w),
                       static_cast<float>(t.rotation.x),
                       static_cast<float>(t.rotation.y),
                       static_cast<float>(t.rotation.z));
  q.normalize();
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = q.toRotationMatrix();
  m(0, 3) = static_cast<float>(t.translation.x);
  m(1, 3) = static_cast<float>(t.translation.y);
  m(2, 3) = static_cast<float>(t.translation.z);
  return m;
}

}  // namespace

class TransformFusion : public rclcpp::Node
{
public:
  TransformFusion() : Node("transform_fusion")
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    body_frame_ = declare_parameter<std::string>("body_frame", "body");
    double rate = declare_parameter<double>("fusion_rate", 50.0);  // upstream: 50 Hz
    // REGRESSION FIX 2026-08-12: was hardcoded "/Odometry", which every mapping
    // launch now remaps onto /odom_lio -- so this had zero publishers and the
    // node silently never saw odometry. See the same fix in global_localization.
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom_lio");

    T_map_to_odom_ = Eigen::Matrix4f::Identity();
    T_lio_base_ = Eigen::Matrix4f::Identity();
    br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    // Relative (no leading '/') so the topic follows the node's namespace and
    // stays remappable per robot; "pose" names the data, "localization" is the
    // subsystem namespace (same shape as Autoware's /localization/...).
    pub_localization_ = create_publisher<nav_msgs::msg::Odometry>("localization/pose", 1);

    auto qos = rclcpp::QoS(1);
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&TransformFusion::cbSaveCurOdom, this, std::placeholders::_1));
    sub_map_to_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/map_to_odom", qos,
      std::bind(&TransformFusion::cbSaveMapToOdom, this, std::placeholders::_1));

    auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&TransformFusion::transformFusion, this));

    RCLCPP_INFO(get_logger(),
                "Transform Fusion Node Inited... broadcasting %s -> %s at %.0f Hz.",
                map_frame_.c_str(), odom_frame_.c_str(), rate);
  }

private:
  void cbSaveCurOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cur_odom_ = msg;
  }

  void cbSaveMapToOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    T_map_to_odom_ = poseToMat(msg->pose.pose);
    corr_cov_ = msg->pose.covariance;   // global_localization sizes it by ICP fitness
    if (!have_correction_) {
      have_correction_ = true;
      RCLCPP_INFO(get_logger(),
                  "First accepted correction received; now broadcasting %s -> %s.",
                  map_frame_.c_str(), odom_frame_.c_str());
    }
  }

  // Resolve <LIO body> -> body_frame_ (e.g. l2lidar_frame_imu -> base_footprint)
  // from TF, once. It comes from the URDF sensor rig and never changes, so a
  // single successful lookup is cached for the life of the node.
  //
  // Why this is needed at all: cur_odom is the LIO's odom -> <its own body>
  // pose, and the LIO body is the IMU/lidar up on the mast, not the footprint
  // between the wheels. Composing map->odom with it yields the LIDAR's pose;
  // publishing that with child_frame_id = base_footprint claims a pose for a
  // frame ~1 m away, and the offset shows up as a phantom arc in x/y whenever
  // the robot turns in place. The extrinsic below is what closes that gap.
  bool ensureExtrinsic(const std::string &lio_body)
  {
    if (have_extrinsic_) return true;

    // A LIO that already reports base_footprint (or reports nothing) needs no
    // correction -- identity is then the right answer, not a missing one.
    if (lio_body.empty() || lio_body == body_frame_) {
      T_lio_base_ = Eigen::Matrix4f::Identity();
      have_extrinsic_ = true;
      RCLCPP_INFO(get_logger(),
                  "Odometry child frame is '%s'; no extrinsic needed.",
                  lio_body.empty() ? "(unset)" : lio_body.c_str());
      return true;
    }

    try {
      auto tf = tf_buffer_->lookupTransform(lio_body, body_frame_, tf2::TimePointZero);
      T_lio_base_ = transformToMat(tf.transform);
      have_extrinsic_ = true;
      RCLCPP_INFO(get_logger(),
                  "Resolved %s -> %s extrinsic ([%.3f %.3f %.3f] m); fused pose "
                  "and twist are now genuinely %s.",
                  lio_body.c_str(), body_frame_.c_str(),
                  T_lio_base_(0, 3), T_lio_base_(1, 3), T_lio_base_(2, 3),
                  body_frame_.c_str());
      return true;
    } catch (const tf2::TransformException &e) {
      auto &clk = *get_clock();
      RCLCPP_WARN_THROTTLE(get_logger(), clk, 5000,
          "Cannot look up %s -> %s (%s) -- withholding the fused pose. The TF "
          "is static from the sensor-rig URDF; check robot_state_publisher is "
          "up. (The %s -> %s broadcast below is unaffected.)",
          lio_body.c_str(), body_frame_.c_str(), e.what(),
          map_frame_.c_str(), odom_frame_.c_str());
      return false;
    }
  }

  void transformFusion()
  {
    Eigen::Matrix4f T;
    nav_msgs::msg::Odometry::SharedPtr cur_odom;
    bool have;
    std::array<double, 36> cov{};
    {
      std::lock_guard<std::mutex> lk(mtx_);
      T = T_map_to_odom_;
      cur_odom = cur_odom_;
      have = have_correction_;
      cov = corr_cov_;
    }

    // Publish NOTHING until global_localization has accepted a match.
    // Broadcasting the identity default would assert map == odom_frame, and
    // odom_frame is the LIO's own gravity-tilted world frame -- so a failed
    // localization would look exactly like a working one: complete TF tree,
    // topics flowing, RViz rendering, and the robot climbing a 90 deg slope
    // because 'map' inherited the sensor mount's tilt. Absent is honest;
    // silently wrong is not.
    if (!have) {
      auto &clk = *get_clock();
      RCLCPP_WARN_THROTTLE(get_logger(), clk, 5000,
          "No accepted correction yet -- not broadcasting %s -> %s. Send an "
          "initial pose, and lower localization_th if ICP keeps rejecting "
          "(current scans may only partly overlap the prior map).",
          map_frame_.c_str(), odom_frame_.c_str());
      return;
    }

    // broadcast map -> odom TF
    Eigen::Matrix3f R = T.block<3, 3>(0, 0);
    Eigen::Quaternionf q(R);
    q.normalize();

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = map_frame_;
    tf.child_frame_id = odom_frame_;
    tf.transform.translation.x = T(0, 3);
    tf.transform.translation.y = T(1, 3);
    tf.transform.translation.z = T(2, 3);
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    br_->sendTransform(tf);

    // publish fused global-localization odometry (map -> body_frame_)
    if (cur_odom && ensureExtrinsic(cur_odom->child_frame_id)) {
      // map->odom * odom-><LIO body> * <LIO body>->base. Upstream stops after
      // the second term and mislabels the result as the base frame.
      Eigen::Matrix4f T_map_base = T * poseToMat(cur_odom->pose.pose) * T_lio_base_;
      Eigen::Matrix3f Rb = T_map_base.block<3, 3>(0, 0);
      Eigen::Quaternionf qb(Rb);
      qb.normalize();

      nav_msgs::msg::Odometry loc;
      loc.header.stamp = cur_odom->header.stamp;   // upstream uses odom stamp
      loc.header.frame_id = map_frame_;
      loc.child_frame_id = body_frame_;
      loc.pose.pose.position.x = T_map_base(0, 3);
      loc.pose.pose.position.y = T_map_base(1, 3);
      loc.pose.pose.position.z = T_map_base(2, 3);
      loc.pose.pose.orientation.x = qb.x();
      loc.pose.pose.orientation.y = qb.y();
      loc.pose.pose.orientation.z = qb.z();
      loc.pose.pose.orientation.w = qb.w();

      // Uncertainty of the map->odom correction that produced this pose,
      // sized from the ICP fitness by global_localization. It does NOT model
      // LIO drift accumulated since that correction, so between ICP updates
      // (2 Hz) the true uncertainty is somewhat larger than reported.
      // Upstream left all 36 entries at zero, which asserts a perfect pose and
      // makes any downstream EKF either ignore every other sensor or blow up
      // on a singular covariance.
      loc.pose.covariance = cov;

      // Odometry semantics: twist is expressed in child_frame_id. Upstream
      // copies the LIO twist verbatim while relabelling the child frame, so
      // the velocities stay in lidar axes under a base_footprint label.
      // Rotate into the base frame and add the lever-arm term:
      //   w_base = R * w_lio
      //   v_base = R * v_lio + w_base x (R * t)
      // where R = R_base<-lio and t = base origin in lidar coords.
      const Eigen::Matrix3f R_bl = T_lio_base_.block<3, 3>(0, 0).transpose();
      const Eigen::Vector3f r_b = R_bl * T_lio_base_.block<3, 1>(0, 3);
      const auto &tw = cur_odom->twist.twist;
      Eigen::Vector3f v_l(static_cast<float>(tw.linear.x),
                          static_cast<float>(tw.linear.y),
                          static_cast<float>(tw.linear.z));
      Eigen::Vector3f w_l(static_cast<float>(tw.angular.x),
                          static_cast<float>(tw.angular.y),
                          static_cast<float>(tw.angular.z));
      const Eigen::Vector3f w_b = R_bl * w_l;
      const Eigen::Vector3f v_b = R_bl * v_l + w_b.cross(r_b);

      loc.twist.twist.linear.x = v_b.x();
      loc.twist.twist.linear.y = v_b.y();
      loc.twist.twist.linear.z = v_b.z();
      loc.twist.twist.angular.x = w_b.x();
      loc.twist.twist.angular.y = w_b.y();
      loc.twist.twist.angular.z = w_b.z();
      loc.twist.covariance = cur_odom->twist.covariance;  // whatever the LIO reports

      pub_localization_->publish(loc);
    }
  }

  std::string map_frame_, odom_frame_, body_frame_, odom_topic_;
  Eigen::Matrix4f T_map_to_odom_;
  Eigen::Matrix4f T_lio_base_;          // <LIO body> -> body_frame_, from TF, cached
  bool have_extrinsic_{false};
  bool have_correction_{false};
  std::array<double, 36> corr_cov_{};   // latest /map_to_odom pose covariance
  nav_msgs::msg::Odometry::SharedPtr cur_odom_;
  std::mutex mtx_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_map_to_odom_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TransformFusion>());
  rclcpp::shutdown();
  return 0;
}
