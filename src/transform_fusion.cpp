// C++ port of FAST_LIO_LOCALIZATION's transform_fusion.py.
//
// Faithful to upstream (HViktorTsoi/FAST_LIO_LOCALIZATION): broadcasts the
// map->odom_lidar TF at FREQ_PUB_LOCALIZATION (50 Hz) from the latest
// /map_to_odom correction (global_localization publishes it at only ~0.5 Hz),
// so TF lookups stay fresh between ICP updates while the LIO's
// odom_lidar->body supplies the smooth high-rate motion underneath. Also
// republishes the fused pose map->body on /localization, copying the odom
// twist and using the odom's own stamp.
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

#include <chrono>
#include <memory>
#include <mutex>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>

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

    T_map_to_odom_ = Eigen::Matrix4f::Identity();
    br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    pub_localization_ = create_publisher<nav_msgs::msg::Odometry>("/localization", 1);

    auto qos = rclcpp::QoS(1);
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", qos,
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
  }

  void transformFusion()
  {
    Eigen::Matrix4f T;
    nav_msgs::msg::Odometry::SharedPtr cur_odom;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      T = T_map_to_odom_;
      cur_odom = cur_odom_;
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

    // publish fused global-localization odometry (map -> body)
    if (cur_odom) {
      Eigen::Matrix4f T_map_base = T * poseToMat(cur_odom->pose.pose);
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
      loc.twist = cur_odom->twist;                 // upstream copies twist
      pub_localization_->publish(loc);
    }
  }

  std::string map_frame_, odom_frame_, body_frame_;
  Eigen::Matrix4f T_map_to_odom_;
  nav_msgs::msg::Odometry::SharedPtr cur_odom_;
  std::mutex mtx_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> br_;
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
