// PicoDet NCNN Detection Engine
//
// This wrapper consumes a PaddleDetection PicoDet model exported via PNNX
// that bakes detection post-processing (decode + NMS + TopK) into the graph.
// ncnn cannot run those tail ops directly, so we:
//   1. Register no-op stubs for the unknown layer types so load_param succeeds.
//   2. Extract intermediate blobs that lie BEFORE the unknown layers:
//        - "317" : per-anchor class scores  [w=num_anchors, h=num_class]
//        - "339" : decoded boxes (xyxy) in network input pixel space
//                  [w=4, h=num_anchors]
//   3. Run threshold + class-aware NMS in C++, then inverse-letterbox.
//
// The model bakes anchors at 320x320, so target_size is fixed at 320.

#ifndef PPDET_PICO_H
#define PPDET_PICO_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <net.h>

struct DetObject
{
    cv::Rect_<float> rect; // x, y, width, height (absolute pixels in original image)
    int label;
    float prob;
};

class PicoDet
{
public:
    PicoDet();
    ~PicoDet();

    int load(const char* parampath, const char* modelpath,
             int num_class_hint, bool use_fp16 = true, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* parampath, const char* modelpath,
             int num_class_hint, bool use_fp16 = true, bool use_gpu = false);

    void set_target_size(int target_size);
    void set_prob_threshold(float threshold);
    void set_nms_threshold(float threshold);

    int get_num_class() const { return num_class; }

    // Set custom label names. If empty, uses built-in COCO/custom labels.
    void set_labels(const std::vector<std::string>& labels) { custom_labels = labels; }

    int detect(const cv::Mat& rgb, std::vector<DetObject>& objects);

    // Anti-spoof: detect if image is from a screen/monitor.
    // Returns true if image appears to be from a screen (moiré pattern detected).
    static bool is_from_screen(const cv::Mat& rgb);

    // Enable/disable anti-spoof check before detection.
    void set_anti_spoof(bool enabled) { anti_spoof_enabled = enabled; }
    bool get_anti_spoof() const { return anti_spoof_enabled; }

    // Draw bounding boxes directly onto an RGB image (modifies in-place).
    void draw_detections(cv::Mat& rgb, const std::vector<DetObject>& objects);

protected:
    ncnn::Net picodet_net;
    int target_size;
    int num_class;
    float prob_threshold;
    float nms_threshold;
    bool anti_spoof_enabled;

    // Model format detection (set during first successful inference)
    // 0 = unknown
    // 1 = format A (in0=meta4, in1=pixels, blobs 317/339) — finetune model with baked postprocess
    // 2 = format B (in0=pixels, in1=meta2, blobs 313/335) — COCO model with baked postprocess
    // 3 = format C (in0=pixels only, out0-7 FPN outputs) — model without postprocess
    int input_format;
    std::string score_blob_name;
    std::string box_blob_name;

    void _detect_input_format();

    // FPN decode for format C
    int _detect_fpn(const ncnn::Mat& in_pad, float scale, int img_w, int img_h, std::vector<DetObject>& objects);

    // Custom labels set from Dart
    std::vector<std::string> custom_labels;
};

#endif // PPDET_PICO_H
