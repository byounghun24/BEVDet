#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/Detection3DArray.h>
#include <geometry_msgs/Quaternion.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      ROS_ERROR("[TensorRT] %s", msg);
    } else if (severity == Severity::kWARNING) {
      ROS_WARN("[TensorRT] %s", msg);
    } else {
      ROS_INFO_STREAM_THROTTLE(5.0, "[TensorRT] " << msg);
    }
  }
};

struct BoxPred {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  float l = 0.f;
  float w = 0.f;
  float h = 0.f;
  float yaw = 0.f;
  float vx = 0.f;
  float vy = 0.f;
  float score = 0.f;
  int label = -1;
};

inline size_t elementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    default:
      return 0;
  }
}

inline int64_t volume(const nvinfer1::Dims& dims) {
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return -1;
    }
    v *= dims.d[i];
  }
  return v;
}

inline bool readBinary(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.good()) {
    return false;
  }
  std::streamsize size = ifs.tellg();
  if (size <= 0) {
    return false;
  }
  out->resize(static_cast<size_t>(size));
  ifs.seekg(0, std::ios::beg);
  return ifs.read(reinterpret_cast<char*>(out->data()), size).good();
}

inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

inline float iouAabbBev(const BoxPred& a, const BoxPred& b) {
  const float ax1 = a.x - a.l * 0.5f;
  const float ax2 = a.x + a.l * 0.5f;
  const float ay1 = a.y - a.w * 0.5f;
  const float ay2 = a.y + a.w * 0.5f;

  const float bx1 = b.x - b.l * 0.5f;
  const float bx2 = b.x + b.l * 0.5f;
  const float by1 = b.y - b.w * 0.5f;
  const float by2 = b.y + b.w * 0.5f;

  const float ix1 = std::max(ax1, bx1);
  const float iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2);
  const float iy2 = std::min(ay2, by2);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float ua = (ax2 - ax1) * (ay2 - ay1);
  const float ub = (bx2 - bx1) * (by2 - by1);
  const float uni = ua + ub - inter;
  if (uni <= 1e-6f) {
    return 0.0f;
  }
  return inter / uni;
}

inline int parseOutputIndex(const std::string& name) {
  const std::string prefix("output_");
  if (name.rfind(prefix, 0) != 0) {
    return -1;
  }
  const std::string tail = name.substr(prefix.size());
  if (tail.empty()) {
    return -1;
  }
  for (char c : tail) {
    if (c < '0' || c > '9') {
      return -1;
    }
  }
  return std::stoi(tail);
}

inline std::vector<int> defaultTaskClassNums() {
  // nuScenes/BEVDet 기본 task 분할
  return {1, 2, 2, 1, 2, 2};
}

}  // namespace

class BevDetTrtNode {
public:
  BevDetTrtNode(ros::NodeHandle& nh) {
    nh.param<std::string>("engine_path", engine_path_, "tensorrt/bevdet_dynamic_fp16_fuse.engine");
    nh.param<std::string>("trtexec_input_dir", trtexec_input_dir_, "tensorrt/trtexec_inputs");
    nh.param<std::string>("trt_plugin_path", trt_plugin_path_, "");
    nh.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
    nh.param<std::string>("camera_info_topic", cam_info_topic_, "/camera/camera_info");
    nh.param<std::string>("output_topic", output_topic_, "/bevdet/detections");

    nh.param<int>("input_width", input_w_, 704);
    nh.param<int>("input_height", input_h_, 256);
    nh.param<int>("num_cams", num_cams_, 6);
    nh.param<int>("crop_x", crop_x_, 0);
    nh.param<int>("crop_y", crop_y_, 0);
    nh.param<int>("crop_width", crop_w_, 0);
    nh.param<int>("crop_height", crop_h_, 0);
    nh.param<bool>("to_rgb", to_rgb_, true);

    nh.param<float>("score_threshold", score_threshold_, 0.1f);
    nh.param<float>("nms_iou_threshold", nms_iou_threshold_, 0.2f);
    nh.param<int>("nms_pre_max", nms_pre_max_, 4096);
    nh.param<int>("nms_post_max", nms_post_max_, 500);

    nh.param<float>("x_start", x_start_, -51.2f);
    nh.param<float>("y_start", y_start_, -51.2f);
    nh.param<float>("x_step", x_step_, 0.8f);
    nh.param<float>("y_step", y_step_, 0.8f);

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

    std::vector<int> task_nums;
    if (!nh.getParam("task_class_nums", task_nums) || task_nums.empty()) {
      task_class_nums_ = defaultTaskClassNums();
    } else {
      task_class_nums_ = task_nums;
    }

    std::vector<double> scale_factors;
    if (nh.getParam("nms_rescale_factor", scale_factors) && !scale_factors.empty()) {
      nms_rescale_factor_.assign(scale_factors.begin(), scale_factors.end());
    }

    if (!initTrt()) {
      throw std::runtime_error("Failed to initialize TensorRT runtime");
    }

    img_sub_ = nh.subscribe(image_topic_, 1, &BevDetTrtNode::onImage, this);
    cam_info_sub_ = nh.subscribe(cam_info_topic_, 1, &BevDetTrtNode::onCamInfo, this);
    det_pub_ = nh.advertise<vision_msgs::Detection3DArray>(output_topic_, 1);

    ROS_INFO("BEVDet TRT node ready. engine=%s", engine_path_.c_str());
  }

  ~BevDetTrtNode() {
    for (void* ptr : bindings_) {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    if (context_ != nullptr) {
      context_->destroy();
      context_ = nullptr;
    }
    if (engine_ != nullptr) {
      engine_->destroy();
      engine_ = nullptr;
    }
    if (runtime_ != nullptr) {
      runtime_->destroy();
      runtime_ = nullptr;
    }
    if (plugin_handle_ != nullptr) {
      dlclose(plugin_handle_);
      plugin_handle_ = nullptr;
    }
  }

private:
  bool maybeLoadPlugin() {
    std::vector<std::string> candidates;
    if (!trt_plugin_path_.empty()) {
      candidates.push_back(trt_plugin_path_);
    }
    candidates.push_back("/home/byounghun/workspace/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so");
    candidates.push_back("/home/byounghun/workspace/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so");

    for (const auto& path : candidates) {
      std::ifstream ifs(path);
      if (!ifs.good()) {
        continue;
      }
      void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (handle == nullptr) {
        ROS_WARN("Failed to dlopen TensorRT plugin %s: %s", path.c_str(), dlerror());
        continue;
      }
      plugin_handle_ = handle;
      ROS_INFO("Loaded TensorRT plugin: %s", path.c_str());
      return true;
    }

    ROS_WARN("No TensorRT plugin library loaded. If engine has custom plugins, deserialization may fail.");
    return true;
  }

  bool initTrt() {
    maybeLoadPlugin();

    std::vector<uint8_t> engine_blob;
    if (!readBinary(engine_path_, &engine_blob)) {
      ROS_ERROR("Failed to read engine file: %s", engine_path_.c_str());
      return false;
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) {
      ROS_ERROR("createInferRuntime failed");
      return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_blob.data(), engine_blob.size(), nullptr);
    if (engine_ == nullptr) {
      ROS_ERROR("deserializeCudaEngine failed for %s", engine_path_.c_str());
      return false;
    }

    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
      ROS_ERROR("createExecutionContext failed");
      return false;
    }

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      ROS_ERROR("cudaStreamCreate failed");
      return false;
    }

    const std::vector<std::string> required_inputs = {
      "img", "ranks_depth", "ranks_feat", "ranks_bev", "interval_starts", "interval_lengths"};

    const int nb_bindings = engine_->getNbBindings();
    bindings_.assign(nb_bindings, nullptr);
    binding_names_.assign(nb_bindings, "");
    binding_dtypes_.assign(nb_bindings, nvinfer1::DataType::kFLOAT);
    binding_dims_.assign(nb_bindings, nvinfer1::Dims{});

    for (int i = 0; i < nb_bindings; ++i) {
      const char* name = engine_->getBindingName(i);
      binding_name_to_idx_[name] = i;
      binding_names_[i] = name;
      if (!engine_->bindingIsInput(i)) {
        output_binding_indices_.push_back(i);
      }
    }

    for (const auto& name : required_inputs) {
      if (binding_name_to_idx_.count(name) == 0) {
        ROS_ERROR("Missing required engine input binding: %s", name.c_str());
        return false;
      }
    }

    if (!loadStaticInputBins()) {
      return false;
    }

    if (!setInputShapes()) {
      return false;
    }

    if (!allocateBindings()) {
      return false;
    }

    if (!uploadStaticInputs()) {
      return false;
    }

    const auto img_idx_it = binding_name_to_idx_.find("img");
    if (img_idx_it == binding_name_to_idx_.end()) {
      ROS_ERROR("Missing img input binding");
      return false;
    }
    img_binding_idx_ = img_idx_it->second;
    img_host_nchw_.resize(static_cast<size_t>(num_cams_) * 3U * static_cast<size_t>(input_h_) * static_cast<size_t>(input_w_));

    if (!prepareOutputOrder()) {
      return false;
    }

    return true;
  }

  bool prepareOutputOrder() {
    std::vector<std::pair<int, int>> ordered;  // (output_number, binding_idx)
    for (int idx : output_binding_indices_) {
      int n = parseOutputIndex(binding_names_[idx]);
      if (n < 0) {
        ROS_WARN("Output binding name is not output_N pattern: %s", binding_names_[idx].c_str());
        continue;
      }
      ordered.emplace_back(n, idx);
    }
    if (ordered.empty()) {
      ROS_ERROR("No output_N style output bindings found.");
      return false;
    }

    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    ordered_output_bindings_.clear();
    for (const auto& p : ordered) {
      ordered_output_bindings_.push_back(p.second);
    }

    if (ordered_output_bindings_.size() % 6 != 0) {
      ROS_WARN("Output count (%zu) is not multiple of 6. Decode may fail.",
               ordered_output_bindings_.size());
    }

    ROS_INFO("Detected %zu output bindings (ordered by output_N).", ordered_output_bindings_.size());
    return true;
  }

  bool loadStaticInputBins() {
    return loadInt32Input("ranks_depth") &&
           loadInt32Input("ranks_feat") &&
           loadInt32Input("ranks_bev") &&
           loadInt32Input("interval_starts") &&
           loadInt32Input("interval_lengths");
  }

  bool loadInt32Input(const std::string& name) {
    const std::string path = trtexec_input_dir_ + "/" + name + ".bin";
    std::vector<uint8_t> raw;
    if (!readBinary(path, &raw)) {
      ROS_ERROR("Failed to read %s", path.c_str());
      return false;
    }
    if ((raw.size() % sizeof(int32_t)) != 0) {
      ROS_ERROR("Invalid int32 bin size for %s", path.c_str());
      return false;
    }

    std::vector<int32_t> values(raw.size() / sizeof(int32_t));
    std::memcpy(values.data(), raw.data(), raw.size());
    static_int_inputs_[name] = std::move(values);
    ROS_INFO("Loaded %s (%zu int32)", path.c_str(), static_int_inputs_[name].size());
    return true;
  }

  bool setInputShapes() {
    if (!setInputShape("img", nvinfer1::Dims4(num_cams_, 3, input_h_, input_w_))) {
      return false;
    }

    if (!setInputShapeFromData("ranks_depth")) return false;
    if (!setInputShapeFromData("ranks_feat")) return false;
    if (!setInputShapeFromData("ranks_bev")) return false;
    if (!setInputShapeFromData("interval_starts")) return false;
    if (!setInputShapeFromData("interval_lengths")) return false;

    return true;
  }

  bool setInputShapeFromData(const std::string& name) {
    const auto it = static_int_inputs_.find(name);
    if (it == static_int_inputs_.end()) {
      ROS_ERROR("Missing host input data for %s", name.c_str());
      return false;
    }
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = static_cast<int>(it->second.size());
    return setInputShape(name, dims);
  }

  bool setInputShape(const std::string& name, const nvinfer1::Dims& dims) {
    auto it = binding_name_to_idx_.find(name);
    if (it == binding_name_to_idx_.end()) {
      ROS_ERROR("Binding not found: %s", name.c_str());
      return false;
    }
    if (!context_->setBindingDimensions(it->second, dims)) {
      ROS_ERROR("setBindingDimensions failed for %s", name.c_str());
      return false;
    }
    return true;
  }

  bool allocateBindings() {
    const int nb_bindings = engine_->getNbBindings();
    for (int i = 0; i < nb_bindings; ++i) {
      const nvinfer1::Dims dims = context_->getBindingDimensions(i);
      const int64_t vol = volume(dims);
      if (vol <= 0) {
        ROS_ERROR("Unresolved binding shape for index=%d name=%s", i, engine_->getBindingName(i));
        return false;
      }

      const nvinfer1::DataType dtype = engine_->getBindingDataType(i);
      const size_t bytes = static_cast<size_t>(vol) * elementSize(dtype);
      if (bytes == 0) {
        ROS_ERROR("Unsupported binding dtype at index=%d name=%s", i, engine_->getBindingName(i));
        return false;
      }

      void* device_ptr = nullptr;
      if (cudaMalloc(&device_ptr, bytes) != cudaSuccess) {
        ROS_ERROR("cudaMalloc failed for binding %s", engine_->getBindingName(i));
        return false;
      }
      bindings_[i] = device_ptr;
      binding_dims_[i] = dims;
      binding_dtypes_[i] = dtype;

      if (!engine_->bindingIsInput(i)) {
        output_host_buffers_[i].resize(bytes);
      }

      ROS_INFO("Binding %-18s idx=%d bytes=%zu", engine_->getBindingName(i), i, bytes);
    }
    return true;
  }

  bool uploadStaticInputs() {
    for (const auto& kv : static_int_inputs_) {
      const auto idx_it = binding_name_to_idx_.find(kv.first);
      if (idx_it == binding_name_to_idx_.end()) {
        ROS_ERROR("Binding missing while uploading: %s", kv.first.c_str());
        return false;
      }
      const int idx = idx_it->second;
      const size_t bytes = kv.second.size() * sizeof(int32_t);
      if (cudaMemcpy(bindings_[idx], kv.second.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ROS_ERROR("cudaMemcpy failed for static input %s", kv.first.c_str());
        return false;
      }
    }
    return true;
  }

  std::vector<float> asFloatVector(int binding_idx) const {
    const auto& raw = output_host_buffers_.at(binding_idx);
    const size_t elem_sz = elementSize(binding_dtypes_[binding_idx]);
    const size_t n = raw.size() / elem_sz;
    std::vector<float> out;
    out.resize(n);

    if (binding_dtypes_[binding_idx] == nvinfer1::DataType::kFLOAT) {
      std::memcpy(out.data(), raw.data(), raw.size());
      return out;
    }
    if (binding_dtypes_[binding_idx] == nvinfer1::DataType::kHALF) {
      const __half* p = reinterpret_cast<const __half*>(raw.data());
      for (size_t i = 0; i < n; ++i) {
        out[i] = __half2float(p[i]);
      }
      return out;
    }

    ROS_WARN_THROTTLE(1.0, "Unsupported output dtype for decode on binding idx=%d", binding_idx);
    std::fill(out.begin(), out.end(), 0.0f);
    return out;
  }

  bool decodeOutputs(std::vector<BoxPred>* out_boxes) {
    out_boxes->clear();
    if (ordered_output_bindings_.empty()) {
      return false;
    }
    if (ordered_output_bindings_.size() % 6 != 0) {
      return false;
    }

    const size_t task_count = ordered_output_bindings_.size() / 6;

    std::vector<int> task_class_nums = task_class_nums_;
    if (task_class_nums.size() != task_count) {
      task_class_nums.clear();
      for (size_t t = 0; t < task_count; ++t) {
        const int heat_idx = ordered_output_bindings_[t * 6 + 5];
        const nvinfer1::Dims d = binding_dims_[heat_idx];
        int c = (d.nbDims >= 2) ? d.d[1] : 1;
        task_class_nums.push_back(std::max(1, c));
      }
      ROS_WARN_THROTTLE(5.0, "task_class_nums size mismatch. Using heatmap channels from engine outputs.");
    }

    int class_total = std::accumulate(task_class_nums.begin(), task_class_nums.end(), 0);
    if (nms_rescale_factor_.size() != static_cast<size_t>(class_total)) {
      nms_rescale_factor_.assign(static_cast<size_t>(class_total), 1.0f);
      ROS_WARN_THROTTLE(5.0, "nms_rescale_factor size mismatch. Fallback to all ones.");
    }

    int class_offset = 0;
    for (size_t t = 0; t < task_count; ++t) {
      const int reg_idx = ordered_output_bindings_[t * 6 + 0];
      const int h_idx = ordered_output_bindings_[t * 6 + 1];
      const int dim_idx = ordered_output_bindings_[t * 6 + 2];
      const int rot_idx = ordered_output_bindings_[t * 6 + 3];
      const int vel_idx = ordered_output_bindings_[t * 6 + 4];
      const int hm_idx = ordered_output_bindings_[t * 6 + 5];

      const nvinfer1::Dims hm_dims = binding_dims_[hm_idx];
      if (hm_dims.nbDims != 4) {
        ROS_WARN_THROTTLE(1.0, "Unexpected heatmap dims for task %zu", t);
        class_offset += task_class_nums[t];
        continue;
      }
      const int c = hm_dims.d[1];
      const int h = hm_dims.d[2];
      const int w = hm_dims.d[3];
      const int map_size = h * w;

      const std::vector<float> reg = asFloatVector(reg_idx);
      const std::vector<float> hei = asFloatVector(h_idx);
      const std::vector<float> dim = asFloatVector(dim_idx);
      const std::vector<float> rot = asFloatVector(rot_idx);
      const std::vector<float> vel = asFloatVector(vel_idx);
      const std::vector<float> hm = asFloatVector(hm_idx);

      std::vector<BoxPred> task_boxes;
      task_boxes.reserve(static_cast<size_t>(nms_pre_max_));

      for (int idx = 0; idx < map_size; ++idx) {
        float best_logit = hm[idx];
        int best_cls = 0;
        for (int ci = 1; ci < c; ++ci) {
          const float v = hm[ci * map_size + idx];
          if (v > best_logit) {
            best_logit = v;
            best_cls = ci;
          }
        }
        const float score = sigmoid(best_logit);
        if (score < score_threshold_) {
          continue;
        }

        const int gy = idx / w;
        const int gx = idx % w;
        const int cls = class_offset + best_cls;
        const float scale = nms_rescale_factor_[cls];

        BoxPred box;
        box.x = (reg[0 * map_size + idx] + static_cast<float>(gx)) * x_step_ + x_start_;
        box.y = (reg[1 * map_size + idx] + static_cast<float>(gy)) * y_step_ + y_start_;
        box.z = hei[idx];
        box.l = std::exp(dim[0 * map_size + idx]) * scale;
        box.w = std::exp(dim[1 * map_size + idx]) * scale;
        box.h = std::exp(dim[2 * map_size + idx]) * scale;
        box.yaw = std::atan2(rot[idx], rot[map_size + idx]);
        box.vx = vel[idx];
        box.vy = vel[map_size + idx];
        box.score = score;
        box.label = cls;
        box.z -= box.h * 0.5f;
        task_boxes.push_back(box);
      }

      std::sort(task_boxes.begin(), task_boxes.end(), [](const BoxPred& a, const BoxPred& b) {
        return a.score > b.score;
      });
      if (task_boxes.size() > static_cast<size_t>(nms_pre_max_)) {
        task_boxes.resize(static_cast<size_t>(nms_pre_max_));
      }

      std::vector<BoxPred> kept;
      kept.reserve(task_boxes.size());
      for (const auto& cand : task_boxes) {
        bool drop = false;
        for (const auto& prev : kept) {
          if (cand.label != prev.label) {
            continue;
          }
          if (iouAabbBev(cand, prev) > nms_iou_threshold_) {
            drop = true;
            break;
          }
        }
        if (!drop) {
          // decode 단계에서 곱했던 scale을 복원
          const float scale_restore = nms_rescale_factor_[cand.label];
          BoxPred b = cand;
          b.l /= scale_restore;
          b.w /= scale_restore;
          b.h /= scale_restore;
          kept.push_back(b);
          if (static_cast<int>(kept.size()) >= nms_post_max_) {
            break;
          }
        }
      }

      out_boxes->insert(out_boxes->end(), kept.begin(), kept.end());
      class_offset += task_class_nums[t];
    }

    std::sort(out_boxes->begin(), out_boxes->end(), [](const BoxPred& a, const BoxPred& b) {
      return a.score > b.score;
    });
    if (out_boxes->size() > static_cast<size_t>(nms_post_max_)) {
      out_boxes->resize(static_cast<size_t>(nms_post_max_));
    }
    return true;
  }

  static geometry_msgs::Quaternion yawToQuaternion(float yaw) {
    geometry_msgs::Quaternion q;
    const float half = yaw * 0.5f;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(half);
    q.w = std::cos(half);
    return q;
  }

  void publishDetections(const std::vector<BoxPred>& boxes, const std_msgs::Header& header) {
    vision_msgs::Detection3DArray msg;
    msg.header = header;

    msg.detections.reserve(boxes.size());
    for (const auto& b : boxes) {
      vision_msgs::Detection3D det;
      det.bbox.center.position.x = b.x;
      det.bbox.center.position.y = b.y;
      det.bbox.center.position.z = b.z + b.h * 0.5f;
      det.bbox.center.orientation = yawToQuaternion(b.yaw);
      det.bbox.size.x = b.l;
      det.bbox.size.y = b.w;
      det.bbox.size.z = b.h;
      msg.detections.push_back(det);
    }

    det_pub_.publish(msg);
  }

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
    if (cropped.cols != input_w_ || cropped.rows != input_h_) {
      crop_resized_buf_.create(input_h_, input_w_, cropped.type());
      cv::resize(cropped, crop_resized_buf_, cv::Size(input_w_, input_h_), 0.0, 0.0, cv::INTER_LINEAR);
      preprocessToNchw(crop_resized_buf_);
    } else {
      preprocessToNchw(cropped);
    }

    const size_t img_bytes = img_host_nchw_.size() * sizeof(float);
    if (cudaMemcpyAsync(bindings_[img_binding_idx_], img_host_nchw_.data(), img_bytes,
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      ROS_ERROR_THROTTLE(1.0, "cudaMemcpyAsync failed for img input");
      return;
    }

    const ros::WallTime t0 = ros::WallTime::now();
    const bool ok = context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!ok) {
      ROS_ERROR_THROTTLE(1.0, "TensorRT enqueueV2 failed");
      return;
    }

    for (int idx : output_binding_indices_) {
      const size_t bytes = output_host_buffers_[idx].size();
      if (bytes == 0) {
        continue;
      }
      if (cudaMemcpyAsync(output_host_buffers_[idx].data(), bindings_[idx], bytes,
                          cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
        ROS_ERROR_THROTTLE(1.0, "cudaMemcpyAsync failed for output idx=%d", idx);
        return;
      }
    }

    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      ROS_ERROR_THROTTLE(1.0, "cudaStreamSynchronize failed");
      return;
    }
    const double dt_ms = (ros::WallTime::now() - t0).toSec() * 1000.0;

    std::vector<BoxPred> boxes;
    if (!decodeOutputs(&boxes)) {
      ROS_ERROR_THROTTLE(1.0, "Failed to decode TRT outputs");
      return;
    }

    ROS_INFO_STREAM_THROTTLE(1.0, "TRT latency=" << dt_ms << " ms, boxes=" << boxes.size());
    publishDetections(boxes, msg->header);
  }

  void preprocessToNchw(const cv::Mat& bgr) {
    last_preprocessed_.create(bgr.rows, bgr.cols, CV_32FC3);
    bgr.convertTo(last_preprocessed_, CV_32FC3);
    if (to_rgb_) {
      cv::cvtColor(last_preprocessed_, last_preprocessed_, cv::COLOR_BGR2RGB);
    }
    cv::subtract(last_preprocessed_, mean_, last_preprocessed_);
    cv::multiply(last_preprocessed_, inv_std_, last_preprocessed_);

    const int h = last_preprocessed_.rows;
    const int w = last_preprocessed_.cols;
    const size_t image_plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    const float* src = reinterpret_cast<const float*>(last_preprocessed_.data);

    for (int cam = 0; cam < num_cams_; ++cam) {
      float* dst_cam = img_host_nchw_.data() + static_cast<size_t>(cam) * 3U * image_plane;
      float* dst_c0 = dst_cam;
      float* dst_c1 = dst_cam + image_plane;
      float* dst_c2 = dst_cam + 2U * image_plane;
      for (size_t i = 0; i < image_plane; ++i) {
        dst_c0[i] = src[i * 3 + 0];
        dst_c1[i] = src[i * 3 + 1];
        dst_c2[i] = src[i * 3 + 2];
      }
    }
  }

  std::string engine_path_;
  std::string trtexec_input_dir_;
  std::string trt_plugin_path_;
  std::string image_topic_;
  std::string cam_info_topic_;
  std::string output_topic_;

  int input_w_ = 0;
  int input_h_ = 0;
  int num_cams_ = 6;
  int crop_x_ = 0;
  int crop_y_ = 0;
  int crop_w_ = 0;
  int crop_h_ = 0;
  bool to_rgb_ = true;

  float score_threshold_ = 0.1f;
  float nms_iou_threshold_ = 0.2f;
  int nms_pre_max_ = 4096;
  int nms_post_max_ = 500;
  float x_start_ = -51.2f;
  float y_start_ = -51.2f;
  float x_step_ = 0.8f;
  float y_step_ = 0.8f;

  std::vector<int> task_class_nums_;
  std::vector<float> nms_rescale_factor_;

  cv::Scalar mean_;
  cv::Scalar inv_std_;
  cv::Mat resized_buf_;
  cv::Mat crop_resized_buf_;
  cv::Mat last_preprocessed_;

  ros::Subscriber img_sub_;
  ros::Subscriber cam_info_sub_;
  ros::Publisher det_pub_;

  sensor_msgs::CameraInfo last_cam_info_;
  bool has_cam_info_ = false;

  TrtLogger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  void* plugin_handle_ = nullptr;

  std::unordered_map<std::string, int> binding_name_to_idx_;
  std::vector<void*> bindings_;
  std::vector<std::string> binding_names_;
  std::vector<nvinfer1::DataType> binding_dtypes_;
  std::vector<nvinfer1::Dims> binding_dims_;

  int img_binding_idx_ = -1;
  std::vector<int> output_binding_indices_;
  std::vector<int> ordered_output_bindings_;
  std::unordered_map<int, std::vector<uint8_t>> output_host_buffers_;

  std::unordered_map<std::string, std::vector<int32_t>> static_int_inputs_;
  std::vector<float> img_host_nchw_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "bevdet_trt_node");
  ros::NodeHandle nh("~");
  try {
    BevDetTrtNode node(nh);
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL("Failed to start node: %s", e.what());
    return 1;
  }
  return 0;
}
