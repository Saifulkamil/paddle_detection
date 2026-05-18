// PicoDet NCNN Detection Engine - Implementation
//
// The shipped model bakes post-processing into the graph (NonMaxSuppression,
// TopK, Gather, F.embedding, Tensor.to, pnnx.Expression). ncnn does not have
// any of those ops natively, so we register no-op stubs for them and extract
// upstream blobs that only depend on standard ncnn layers.
//
// Intermediate blobs we read:
//   "317" : class scores      [w = num_anchors, h = num_class]
//   "339" : decoded boxes xyxy [w = 4,           h = num_anchors]
//           (already in 320x320 input pixel coords)
//
// We then run threshold + class-aware NMS, and inverse the letterbox.

#include "ppdet_pico.h"

#include "cpu.h"
#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <android/log.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#define TAG "PicoDet"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================================
// No-op stub layer used to satisfy load_param for unknown ops.
// ncnn evaluates lazily, so as long as we never extract a downstream blob,
// these forwards are never called. We still implement them defensively.
// ============================================================================

namespace {

class NoopLayer : public ncnn::Layer
{
public:
    NoopLayer()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& /*pd*/) override
    {
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& /*bottom_blobs*/,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const override
    {
        // Produce empty mats so downstream code that does extract on these
        // (we never do, but defensively) doesn't dereference null.
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            top_blobs[i] = ncnn::Mat(1, (size_t)4u, opt.blob_allocator);
        }
        return 0;
    }
};

::ncnn::Layer* noop_layer_creator(void* /*userdata*/)
{
    return new NoopLayer;
}

// ----- helpers for NMS -----

inline float intersection_area(const DetObject& a, const DetObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent(std::vector<DetObject>& objs, int left, int right)
{
    int i = left;
    int j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j)
    {
        while (objs[i].prob > p) i++;
        while (objs[j].prob < p) j--;
        if (i <= j)
        {
            std::swap(objs[i], objs[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent(objs, left, j);
    if (i < right) qsort_descent(objs, i, right);
}

void qsort_descent(std::vector<DetObject>& objs)
{
    if (objs.empty()) return;
    qsort_descent(objs, 0, (int)objs.size() - 1);
}

void nms_class_aware(const std::vector<DetObject>& objs,
                     std::vector<int>& picked,
                     float nms_threshold)
{
    picked.clear();
    const int n = (int)objs.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objs[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const DetObject& a = objs[i];
        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const DetObject& b = objs[picked[j]];
            if (a.label != b.label) continue;
            float inter = intersection_area(a, b);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (uni > 0 && inter / uni > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

// Read a value from a 2D ncnn::Mat treated as [h rows, w cols], with row()
// returning the pointer to row r.
inline float at2d(const ncnn::Mat& m, int r, int c)
{
    return m.row(r)[c];
}

} // namespace

// ============================================================================
// PicoDet
// ============================================================================

PicoDet::PicoDet()
    : target_size(320),
      num_class(1),
      prob_threshold(0.4f),
      nms_threshold(0.5f),
      anti_spoof_enabled(false),
      input_format(0)
{
}

PicoDet::~PicoDet()
{
    picodet_net.clear();
}

static void register_stubs(ncnn::Net& net)
{
    // Each unknown type gets the same noop creator. They appear only in the
    // post-processing tail of the graph and we never extract any blob beyond
    // the intermediate ones, so they are never invoked.
    const char* unknown_types[] = {
        "pnnx.Expression",
        "NonMaxSuppression",
        "TopK",
        "Gather",
        "F.embedding",
        "Tensor.to",
        "NonZero",
        "ExpandDims",
    };
    for (const char* t : unknown_types)
    {
        int ret = net.register_custom_layer(t, noop_layer_creator);
        if (ret != 0)
        {
            LOGE("register_custom_layer(%s) failed: %d", t, ret);
        }
    }
}

int PicoDet::load(const char* parampath, const char* modelpath,
                  int num_class_hint, bool use_fp16, bool use_gpu)
{
    picodet_net.clear();
    num_class = std::max(1, num_class_hint);
    input_format = 0;
    score_blob_name.clear();
    box_blob_name.clear();

    picodet_net.opt.use_fp16_packed = use_fp16;
    picodet_net.opt.use_fp16_storage = use_fp16;
    picodet_net.opt.use_fp16_arithmetic = use_fp16;
    picodet_net.opt.num_threads = ncnn::get_big_cpu_count();
#if NCNN_VULKAN
    picodet_net.opt.use_vulkan_compute = use_gpu;
#endif

    register_stubs(picodet_net);

    int ret = picodet_net.load_param(parampath);
    if (ret != 0)
    {
        LOGE("load_param failed: %s, ret=%d", parampath, ret);
        return -1;
    }
    ret = picodet_net.load_model(modelpath);
    if (ret != 0)
    {
        LOGE("load_model failed: %s, ret=%d", modelpath, ret);
        return -2;
    }

    LOGD("Model loaded (file): num_class_hint=%d, target_size=%d", num_class, target_size);
    // Auto-detect input format from graph structure
    _detect_input_format();
    return 0;
}

int PicoDet::load(AAssetManager* mgr, const char* parampath, const char* modelpath,
                  int num_class_hint, bool use_fp16, bool use_gpu)
{
    picodet_net.clear();
    num_class = std::max(1, num_class_hint);
    input_format = 0;
    score_blob_name.clear();
    box_blob_name.clear();

    picodet_net.opt.use_fp16_packed = use_fp16;
    picodet_net.opt.use_fp16_storage = use_fp16;
    picodet_net.opt.use_fp16_arithmetic = use_fp16;
    picodet_net.opt.num_threads = ncnn::get_big_cpu_count();
#if NCNN_VULKAN
    picodet_net.opt.use_vulkan_compute = use_gpu;
#endif

    register_stubs(picodet_net);

    int ret = picodet_net.load_param(mgr, parampath);
    if (ret != 0)
    {
        LOGE("load_param (asset) failed: %s, ret=%d", parampath, ret);
        return -1;
    }
    ret = picodet_net.load_model(mgr, modelpath);
    if (ret != 0)
    {
        LOGE("load_model (asset) failed: %s, ret=%d", modelpath, ret);
        return -2;
    }

    LOGD("Model loaded (asset): num_class_hint=%d, target_size=%d", num_class, target_size);
    // Auto-detect input format from graph structure
    _detect_input_format();
    return 0;
}

void PicoDet::set_target_size(int /*_target_size*/)
{
    // The shipped model bakes 320x320 anchors. Forcing 320 keeps decoding correct.
    target_size = 320;
}

void PicoDet::set_prob_threshold(float t) { prob_threshold = t; }
void PicoDet::set_nms_threshold(float t)  { nms_threshold = t; }

void PicoDet::_detect_input_format()
{
    const std::vector<ncnn::Layer*>& layers = picodet_net.layers();

    // Count Input layers
    int input_count = 0;
    for (size_t i = 0; i < layers.size() && i < 5; i++) {
        if (layers[i]->bottoms.empty() && layers[i]->tops.size() == 1)
            input_count++;
    }

    if (input_count == 1)
    {
        // Single input → Format C (no postprocess, FPN decode)
        input_format = 3;
        // Detect num_class by quick-extracting out0 shape
        ncnn::Mat dummy_in(target_size, target_size, 3, (size_t)4u);
        dummy_in.fill(0.f);
        ncnn::Extractor ex = picodet_net.create_extractor();
        ex.input("in0", dummy_in);
        ncnn::Mat cls0;
        if (ex.extract("out0", cls0) == 0 && cls0.h > 0 && cls0.w > 0) {
            // After Permute, shape is [h=num_class, w=num_anchors] or vice versa
            // num_class is always smaller than num_anchors
            num_class = std::min(cls0.h, cls0.w);
            LOGD("_detect_input_format: Format C, num_class=%d from out0 shape (h=%d w=%d)", num_class, cls0.h, cls0.w);
        } else {
            LOGD("_detect_input_format: Format C, could not detect num_class, keeping %d", num_class);
        }
        return;
    }

    // 2 inputs — find first non-Input layer
    size_t first_real = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        if (!layers[i]->bottoms.empty()) { first_real = i; break; }
    }
    if (first_real == 0 || layers[first_real]->bottoms.empty()) {
        input_format = 1; score_blob_name = "317"; box_blob_name = "339"; return;
    }

    int first_input_blob = layers[first_real]->bottoms[0];
    if (first_input_blob == 0) {
        input_format = 2; score_blob_name = "313"; box_blob_name = "335";
        LOGD("_detect_input_format: Format B (in0=pixels, in1=meta2)");
    } else {
        input_format = 1; score_blob_name = "317"; box_blob_name = "339";
        LOGD("_detect_input_format: Format A (in0=meta4, in1=pixels)");
    }

    // Detect num_class via dummy inference for Format A/B
    {
        ncnn::Mat dummy_pixels(target_size, target_size, 3, (size_t)4u);
        dummy_pixels.fill(0.f);
        ncnn::Mat dummy_meta4(4);
        { float* p = (float*)dummy_meta4.data; p[0] = 320; p[1] = 320; p[2] = 1.0f; p[3] = 1.0f; }
        ncnn::Mat dummy_meta2(2);
        { float* p = (float*)dummy_meta2.data; p[0] = 1.0f; p[1] = 1.0f; }

        ncnn::Extractor ex = picodet_net.create_extractor();
        if (input_format == 2) { ex.input("in0", dummy_pixels); ex.input("in1", dummy_meta2); }
        else { ex.input("in0", dummy_meta4); ex.input("in1", dummy_pixels); }

        ncnn::Mat scores;
        if (ex.extract(score_blob_name.c_str(), scores) == 0 && scores.w > 0 && scores.h > 0) {
            num_class = std::min(scores.w, scores.h);
            LOGD("_detect_input_format: num_class=%d from scores shape (w=%d h=%d)", num_class, scores.w, scores.h);
        }
    }
}

// ============================================================================
// FPN decode for format C (model without postprocess)
// Outputs: out0-3 = cls [num_class, anchors], out4-7 = dis [32, anchors]
// Strides: 8, 16, 32, 64. reg_max=7 (32 = 4*8 DFL bins)
// ============================================================================

int PicoDet::_detect_fpn(const ncnn::Mat& in_pad, float scale, int img_w, int img_h, std::vector<DetObject>& objects)
{
    static const int strides[] = {8, 16, 32, 64};
    static const char* cls_names[] = {"out0", "out1", "out2", "out3"};
    static const char* dis_names[] = {"out4", "out5", "out6", "out7"};
    const int reg_max = 7;
    const int num_strides = 4;

    ncnn::Extractor ex = picodet_net.create_extractor();
    ex.input("in0", in_pad);

    std::vector<DetObject> proposals;
    proposals.reserve(256);

    for (int s = 0; s < num_strides; s++)
    {
        ncnn::Mat cls_mat, dis_mat;
        if (ex.extract(cls_names[s], cls_mat) != 0) { LOGE("FPN: extract %s failed", cls_names[s]); continue; }
        if (ex.extract(dis_names[s], dis_mat) != 0) { LOGE("FPN: extract %s failed", dis_names[s]); continue; }

        // After Permute: shape could be [h=num_class, w=anchors] or [h=anchors, w=num_class]
        // num_class is always smaller dimension
        int nc, num_anchors_stride;
        if (cls_mat.h <= cls_mat.w) {
            nc = cls_mat.h;
            num_anchors_stride = cls_mat.w;
        } else {
            nc = cls_mat.w;
            num_anchors_stride = cls_mat.h;
        }
        const int feat_w = target_size / strides[s];
        const int feat_h = target_size / strides[s];
        const bool cls_row_is_class = (cls_mat.h == nc); // true if row=class, col=anchor

        // Update num_class from first stride
        if (s == 0) num_class = nc;

        for (int a = 0; a < num_anchors_stride; a++)
        {
            // Find best class
            int best_label = 0;
            float best_score = -FLT_MAX;
            for (int k = 0; k < nc; k++)
            {
                float score = cls_row_is_class ? cls_mat.row(k)[a] : cls_mat.row(a)[k];
                if (score > best_score) { best_score = score; best_label = k; }
            }
            if (best_score < prob_threshold) continue;

            // DFL decode: softmax over reg_max+1 bins per side
            // dis_mat: [h=32, w=anchors] or [h=anchors, w=32]
            const bool dis_row_is_bin = (dis_mat.h == 4 * (reg_max + 1));
            float dist[4];
            for (int side = 0; side < 4; side++)
            {
                float maxv = -FLT_MAX;
                for (int b = 0; b <= reg_max; b++) {
                    float v = dis_row_is_bin ? dis_mat.row(side * (reg_max + 1) + b)[a] : dis_mat.row(a)[side * (reg_max + 1) + b];
                    if (v > maxv) maxv = v;
                }
                float exp_sum = 0.f;
                for (int b = 0; b <= reg_max; b++) {
                    float v = dis_row_is_bin ? dis_mat.row(side * (reg_max + 1) + b)[a] : dis_mat.row(a)[side * (reg_max + 1) + b];
                    exp_sum += std::exp(v - maxv);
                }
                float expectation = 0.f;
                for (int b = 0; b <= reg_max; b++) {
                    float v = dis_row_is_bin ? dis_mat.row(side * (reg_max + 1) + b)[a] : dis_mat.row(a)[side * (reg_max + 1) + b];
                    expectation += b * (std::exp(v - maxv) / exp_sum);
                }
                dist[side] = expectation;
            }

            int gx = a % feat_w;
            int gy = a / feat_w;
            float cx = (gx + 0.5f) * strides[s];
            float cy = (gy + 0.5f) * strides[s];

            float x1 = cx - dist[0] * strides[s];
            float y1 = cy - dist[1] * strides[s];
            float x2 = cx + dist[2] * strides[s];
            float y2 = cy + dist[3] * strides[s];

            DetObject obj;
            obj.label = best_label;
            obj.prob = best_score;
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            if (obj.rect.width > 0 && obj.rect.height > 0)
                proposals.push_back(obj);
        }
    }

    // NMS
    qsort_descent(proposals);
    std::vector<int> picked;
    nms_class_aware(proposals, picked, nms_threshold);

    // Inverse letterbox
    objects.reserve(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        DetObject obj = proposals[picked[i]];
        float x1 = obj.rect.x / scale;
        float y1 = obj.rect.y / scale;
        float x2 = (obj.rect.x + obj.rect.width) / scale;
        float y2 = (obj.rect.y + obj.rect.height) / scale;
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(img_w - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(img_h - 1)), 0.f);
        obj.rect.x = x1; obj.rect.y = y1;
        obj.rect.width = x2 - x1; obj.rect.height = y2 - y1;
        if (obj.rect.width > 0 && obj.rect.height > 0)
            objects.push_back(obj);
    }

    LOGD("FPN detect: proposals=%d kept=%d", (int)proposals.size(), (int)objects.size());
    return 0;
}

int PicoDet::detect(const cv::Mat& rgb, std::vector<DetObject>& objects)
{
    objects.clear();

    if (anti_spoof_enabled && is_from_screen(rgb))
    {
        LOGD("detect: anti-spoof triggered");
        return 1;
    }

    const int img_w = rgb.cols;
    const int img_h = rgb.rows;

    // Letterbox
    int w = img_w, h = img_h;
    float scale;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = (int)(img_h * scale); }
    else { scale = (float)target_size / h; h = target_size; w = (int)(img_w * scale); }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
    ncnn::Mat in_pad;
    int wpad = target_size - w, hpad = target_size - h;
    if (wpad > 0 || hpad > 0)
        ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 0.f);
    else
        in_pad = in;

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.017125f, 0.017507f, 0.017429f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    // Format C: FPN decode (single input, no postprocess)
    if (input_format == 3)
    {
        return _detect_fpn(in_pad, scale, img_w, img_h, objects);
    }

    // Format A/B: extract intermediate blobs
    ncnn::Mat meta4(4);
    { float* p = (float*)meta4.data; p[0] = (float)target_size; p[1] = (float)target_size; p[2] = scale; p[3] = scale; }
    ncnn::Mat meta2(2);
    { float* p = (float*)meta2.data; p[0] = scale; p[1] = scale; }

    ncnn::Extractor ex = picodet_net.create_extractor();
    if (input_format == 2) { ex.input("in0", in_pad); ex.input("in1", meta2); }
    else { ex.input("in0", meta4); ex.input("in1", in_pad); }

    ncnn::Mat scores, boxes;
    int r1 = ex.extract(score_blob_name.c_str(), scores);
    int r2 = ex.extract(box_blob_name.c_str(), boxes);

    if (r1 != 0 || r2 != 0 || scores.empty() || boxes.empty() || scores.w == 0 || boxes.w == 0)
    {
        LOGE("extract failed: r1=%d r2=%d format=%d", r1, r2, input_format);
        return -1;
    }

    // Determine which dim is num_class vs num_anchors
    // num_class is always much smaller than num_anchors
    int num_anchors, nc;
    bool scores_transposed; // true if scores layout is [w=nc, h=anchors] instead of [w=anchors, h=nc]
    if (scores.w > scores.h) {
        num_anchors = scores.w;
        nc = scores.h;
        scores_transposed = false; // row = class, col = anchor
    } else {
        num_anchors = scores.h;
        nc = scores.w;
        scores_transposed = true; // row = anchor, col = class
    }
    num_class = nc;

    // Verify boxes shape
    bool boxes_match = false;
    bool boxes_transposed = false;
    if (boxes.w == 4 && boxes.h == num_anchors) { boxes_match = true; boxes_transposed = false; }
    else if (boxes.h == 4 && boxes.w == num_anchors) { boxes_match = true; boxes_transposed = true; }
    if (!boxes_match) {
        LOGE("boxes shape mismatch: w=%d h=%d (anchors=%d)", boxes.w, boxes.h, num_anchors);
        return -3;
    }

    LOGD("scores: w=%d h=%d → nc=%d anchors=%d transposed=%d", scores.w, scores.h, nc, num_anchors, scores_transposed);

    std::vector<DetObject> proposals;
    proposals.reserve(256);
    for (int a = 0; a < num_anchors; a++)
    {
        int best_label = 0; float best_score = -FLT_MAX;
        for (int k = 0; k < nc; k++) {
            float s = scores_transposed ? scores.row(a)[k] : scores.row(k)[a];
            if (s > best_score) { best_score = s; best_label = k; }
        }
        if (best_score < prob_threshold) continue;
        const float* row = boxes_transposed ? nullptr : boxes.row(a);
        float x1, y1, x2, y2;
        if (boxes_transposed) {
            x1 = boxes.row(0)[a]; y1 = boxes.row(1)[a]; x2 = boxes.row(2)[a]; y2 = boxes.row(3)[a];
        } else {
            x1 = row[0]; y1 = row[1]; x2 = row[2]; y2 = row[3];
        }
        if (x2 <= x1 || y2 <= y1) continue;
        DetObject obj; obj.label = best_label; obj.prob = best_score;
        obj.rect.x = x1; obj.rect.y = y1; obj.rect.width = x2-x1; obj.rect.height = y2-y1;
        proposals.push_back(obj);
    }

    qsort_descent(proposals);
    std::vector<int> picked;
    nms_class_aware(proposals, picked, nms_threshold);

    objects.reserve(picked.size());
    for (size_t i = 0; i < picked.size(); i++) {
        DetObject obj = proposals[picked[i]];
        float x1 = obj.rect.x / scale, y1 = obj.rect.y / scale;
        float x2 = (obj.rect.x + obj.rect.width) / scale, y2 = (obj.rect.y + obj.rect.height) / scale;
        x1 = std::max(std::min(x1, (float)(img_w-1)), 0.f); y1 = std::max(std::min(y1, (float)(img_h-1)), 0.f);
        x2 = std::max(std::min(x2, (float)(img_w-1)), 0.f); y2 = std::max(std::min(y2, (float)(img_h-1)), 0.f);
        obj.rect.x = x1; obj.rect.y = y1; obj.rect.width = x2-x1; obj.rect.height = y2-y1;
        if (obj.rect.width > 0 && obj.rect.height > 0) objects.push_back(obj);
    }

    LOGD("detect: anchors=%d proposals=%d kept=%d", num_anchors, (int)proposals.size(), (int)objects.size());
    return 0;
}


// ============================================================================
// Anti-spoof: Detect if image is from a screen/monitor
// Uses moiré pattern detection via FFT frequency analysis
// ============================================================================

bool PicoDet::is_from_screen(const cv::Mat& rgb)
{
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

    // Resize to fixed size for consistent analysis
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(256, 256));

    // Convert to float for DFT
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F);

    // Apply DFT
    cv::Mat dft_result;
    cv::dft(floatImg, dft_result, cv::DFT_COMPLEX_OUTPUT);

    // Shift zero-frequency to center
    int cx = dft_result.cols / 2;
    int cy = dft_result.rows / 2;

    // Split into real and imaginary
    std::vector<cv::Mat> planes;
    cv::split(dft_result, planes);

    // Compute magnitude
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    // Log scale
    magnitude += cv::Scalar::all(1);
    cv::log(magnitude, magnitude);

    // Normalize
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

    // Analyze high-frequency content
    // Screen moiré creates peaks in mid-to-high frequency bands
    // Measure energy in high-frequency ring (exclude DC center and very high noise)
    float high_freq_energy = 0;
    float mid_freq_energy = 0;
    float total_energy = 0;
    int high_count = 0;
    int mid_count = 0;
    int total_count = 0;

    for (int y = 0; y < magnitude.rows; y++)
    {
        for (int x = 0; x < magnitude.cols; x++)
        {
            float dist = std::sqrt((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
            float val = magnitude.at<float>(y, x);
            total_energy += val;
            total_count++;

            // Mid frequency band (20-60% of max radius)
            if (dist > cx * 0.2f && dist < cx * 0.6f)
            {
                mid_freq_energy += val;
                mid_count++;
            }
            // High frequency band (60-90% of max radius)
            if (dist > cx * 0.6f && dist < cx * 0.9f)
            {
                high_freq_energy += val;
                high_count++;
            }
        }
    }

    // Compute ratios
    float avg_mid = mid_count > 0 ? mid_freq_energy / mid_count : 0;
    float avg_high = high_count > 0 ? high_freq_energy / high_count : 0;
    float avg_total = total_count > 0 ? total_energy / total_count : 0;

    // Moiré pattern indicator: high ratio of mid+high frequency to total
    // Real-world images have most energy in low frequencies
    // Screen images have elevated mid/high frequency due to pixel grid
    float screen_score = (avg_mid + avg_high) / (avg_total + 1e-6f);

    // Also check for periodic peaks (moiré is periodic)
    // Count pixels above threshold in high-freq band
    int peak_count = 0;
    float peak_threshold = avg_high * 2.0f;
    for (int y = 0; y < magnitude.rows; y++)
    {
        for (int x = 0; x < magnitude.cols; x++)
        {
            float dist = std::sqrt((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
            if (dist > cx * 0.4f && dist < cx * 0.85f)
            {
                if (magnitude.at<float>(y, x) > peak_threshold)
                    peak_count++;
            }
        }
    }

    float peak_ratio = (float)peak_count / (float)(magnitude.rows * magnitude.cols);

    LOGD("anti-spoof: screen_score=%.4f peak_ratio=%.4f", screen_score, peak_ratio);

    // Thresholds (tuned empirically)
    // screen_score > 1.5 OR peak_ratio > 0.05 → likely from screen
    bool is_screen = (screen_score > 1.5f) || (peak_ratio > 0.05f);

    return is_screen;
}

// ============================================================================
// Draw detections onto RGB image
// ============================================================================

void PicoDet::draw_detections(cv::Mat& rgb, const std::vector<DetObject>& objects)
{
    // COCO 80 class names (fallback)
    static const char* coco_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const DetObject& obj = objects[i];

        // Determine label name
        char name_buf[64];
        const char* name;

        if (!custom_labels.empty() && obj.label >= 0 && obj.label < (int)custom_labels.size())
        {
            // Use user-provided labels
            name = custom_labels[obj.label].c_str();
        }
        else if (num_class == 80 && obj.label >= 0 && obj.label < 80)
        {
            // COCO model
            name = coco_names[obj.label];
        }
        else
        {
            // Generic: show "C{index}"
            snprintf(name_buf, sizeof(name_buf), "C%d", obj.label);
            name = name_buf;
        }

        // Generate color from label index (deterministic)
        int r = (obj.label * 67 + 50) % 200 + 55;
        int g = (obj.label * 113 + 100) % 200 + 55;
        int b = (obj.label * 37 + 150) % 200 + 55;
        cv::Scalar color(r, g, b);

        // Draw rectangle
        int x1 = (int)obj.rect.x;
        int y1 = (int)obj.rect.y;
        int x2 = (int)(obj.rect.x + obj.rect.width);
        int y2 = (int)(obj.rect.y + obj.rect.height);

        cv::rectangle(rgb, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Draw label background + text
        char text[64];
        snprintf(text, sizeof(text), "%s %.0f%%", name, obj.prob * 100.f);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int label_y = std::max(y1 - label_size.height - 4, 0);
        cv::rectangle(rgb,
                      cv::Point(x1, label_y),
                      cv::Point(x1 + label_size.width + 4, label_y + label_size.height + baseLine + 4),
                      color, -1); // filled

        cv::putText(rgb, text,
                    cv::Point(x1 + 2, label_y + label_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}
