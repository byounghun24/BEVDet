#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/Detection3DArray.h>

#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// TODO: include TensorRT headers and your engine wrapper
// #include <NvInfer.h>

class BevDetTrtNode {
public:
  BevDetTrtNode(ros::NodeHandle& nh) {
    nh.param<std::string>("engine_path", engine_path_, "bevdet.engine");
    nh.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
    nh.param<std::string>("camera_info_topic", cam_info_topic_, "/camera/camera_info");
    nh.param<std::string>("output_topic", output_topic_, "/bevdet/detections");
    nh.param<int>("input_width", input_w_, 704);
    nh.param<int>("input_height", input_h_, 256);
    nh.param<int>("crop_x", crop_x_, 0);
    nh.param<int>("crop_y", crop_y_, 0);
    nh.param<int>("crop_width", crop_w_, 0);
    nh.param<int>("crop_height", crop_h_, 0);
    nh.param<bool>("to_rgb", to_rgb_, true);

    std::vector<double> mean_vec{123.675, 116.28, 103.53};
    std::vector<double> std_vec{58.395, 57.12, 57.375};
    nh.getParam("mean", mean_vec);
    nh.getParam("std", std_vec);
    if (mean_vec.size() != 3) {
      mean_vec = {123.675, 116.28, 103.53};
    }
    if (std_vec.size() != 3) {
      std_vec = {58.395, 57.12, 57.375};
    }
    if (to_rgb_) {
      mean_ = cv::Scalar(mean_vec[0], mean_vec[1], mean_vec[2]);
      inv_std_ = cv::Scalar(1.0 / std_vec[0], 1.0 / std_vec[1], 1.0 / std_vec[2]);
    } else {
      mean_ = cv::Scalar(mean_vec[2], mean_vec[1], mean_vec[0]);
      inv_std_ = cv::Scalar(1.0 / std_vec[2], 1.0 / std_vec[1], 1.0 / std_vec[0]);
    }

    // TODO: load TensorRT engine here

    img_sub_ = nh.subscribe(image_topic_, 1, &BevDetTrtNode::onImage, this);
    cam_info_sub_ = nh.subscribe(cam_info_topic_, 1, &BevDetTrtNode::onCamInfo, this);
    det_pub_ = nh.advertise<vision_msgs::Detection3DArray>(output_topic_, 1);
  }

private:
  void onCamInfo(const sensor_msgs::CameraInfoConstPtr& msg) {
    last_cam_info_ = *msg;
    has_cam_info_ = true;
  }

  void onImage(const sensor_msgs::ImageConstPtr& msg) {
    if (!has_cam_info_) {
      ROS_WARN_THROTTLE(5.0, "Waiting for camera info...");
      return;
    }

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge error: %s", e.what());
      return;
    }

    const cv::Mat& src = cv_ptr->image;
    cv::Mat resized = src;
    if (input_w_ > 0 && input_h_ > 0 &&
        (src.cols != input_w_ || src.rows != input_h_)) {
      resized_buf_.create(input_h_, input_w_, src.type());
      cv::resize(src, resized_buf_, cv::Size(input_w_, input_h_), 0.0, 0.0, cv::INTER_LINEAR);
      resized = resized_buf_;
    }

    cv::Rect roi(0, 0, resized.cols, resized.rows);
    if (crop_w_ > 0 && crop_h_ > 0) {
      const int x = std::max(0, crop_x_);
      const int y = std::max(0, crop_y_);
      const int w = std::min(crop_w_, resized.cols - x);
      const int h = std::min(crop_h_, resized.rows - y);
      if (w > 0 && h > 0) {
        roi = cv::Rect(x, y, w, h);
      }
    }

    const cv::Mat cropped = resized(roi);
    last_preprocessed_.create(cropped.rows, cropped.cols, CV_32FC3);
    cropped.convertTo(last_preprocessed_, CV_32FC3);
    if (to_rgb_) {
      cv::cvtColor(last_preprocessed_, last_preprocessed_, cv::COLOR_BGR2RGB);
    }
    cv::subtract(last_preprocessed_, mean_, last_preprocessed_);
    cv::multiply(last_preprocessed_, inv_std_, last_preprocessed_);
    // TODO: build TRT inputs (images, intrinsics, etc.)
    // TODO: run TRT inference
    // TODO: postprocess to Detection3DArray

    vision_msgs::Detection3DArray detections;
    detections.header = msg->header;
    det_pub_.publish(detections);
  }

  std::string engine_path_;
  std::string image_topic_;
  std::string cam_info_topic_;
  std::string output_topic_;
  int input_w_ = 0;
  int input_h_ = 0;
  int crop_x_ = 0;
  int crop_y_ = 0;
  int crop_w_ = 0;
  int crop_h_ = 0;
  bool to_rgb_ = true;
  cv::Scalar mean_;
  cv::Scalar inv_std_;
  cv::Mat resized_buf_;

  ros::Subscriber img_sub_;
  ros::Subscriber cam_info_sub_;
  ros::Publisher det_pub_;

  sensor_msgs::CameraInfo last_cam_info_;
  bool has_cam_info_ = false;
  cv::Mat last_preprocessed_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "bevdet_trt_node");
  ros::NodeHandle nh("~");
  BevDetTrtNode node(nh);
  ros::spin();
  return 0;
}
