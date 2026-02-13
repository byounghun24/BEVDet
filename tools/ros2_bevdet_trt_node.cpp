#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// TODO: include TensorRT headers and your engine wrapper
// #include <NvInfer.h>

class BevDetTrtNode : public rclcpp::Node {
public:
  BevDetTrtNode() : Node("bevdet_trt_node") {
    engine_path_ = this->declare_parameter<std::string>("engine_path", "bevdet.engine");
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    cam_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "/bevdet/detections");

    // TODO: load TensorRT engine here

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&BevDetTrtNode::onImage, this, std::placeholders::_1));

    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      cam_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&BevDetTrtNode::onCamInfo, this, std::placeholders::_1));

    det_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(output_topic_, 1);
  }

private:
  void onCamInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    last_cam_info_ = *msg;
    has_cam_info_ = true;
  }

  void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!has_cam_info_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Waiting for camera info...");
      return;
    }

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
      return;
    }

    // TODO: preprocess image (resize/crop/normalize)
    // TODO: build TRT inputs (images, intrinsics, etc.)
    // TODO: run TRT inference
    // TODO: postprocess to Detection3DArray

    vision_msgs::msg::Detection3DArray detections;
    detections.header = msg->header;
    det_pub_->publish(detections);
  }

  std::string engine_path_;
  std::string image_topic_;
  std::string cam_info_topic_;
  std::string output_topic_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr det_pub_;

  sensor_msgs::msg::CameraInfo last_cam_info_;
  bool has_cam_info_ = false;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BevDetTrtNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
