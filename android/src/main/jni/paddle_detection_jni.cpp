// PicoDet JNI Bridge
// Connects Kotlin PaddleDetectionPlugin <-> C++ PicoDet engine
// Thread safety: ncnn::Mutex guards all access to g_picodet
// GPU lifecycle: create_gpu_instance in JNI_OnLoad, destroy in JNI_OnUnload

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>

#include "ppdet_pico.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define TAG "PaddleDetJNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================================
// Global state (thread-safe via ncnn::Mutex)
// ============================================================================

static PicoDet* g_picodet = nullptr;
static ncnn::Mutex g_lock;

// ============================================================================
// GPU lifecycle
// ============================================================================

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    LOGD("JNI_OnLoad");
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    LOGD("JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(g_lock);
        delete g_picodet;
        g_picodet = nullptr;
    }

    ncnn::destroy_gpu_instance();
}

// ============================================================================
// Helper: convert detection results to Java float[][]
// Each row: [label, prob, x, y, width, height]
// ============================================================================

static jobjectArray detections_to_java(JNIEnv* env, const std::vector<DetObject>& objects)
{
    jclass floatArrayClass = env->FindClass("[F");
    jobjectArray result = env->NewObjectArray(objects.size(), floatArrayClass, nullptr);

    for (int i = 0; i < (int)objects.size(); i++)
    {
        const DetObject& obj = objects[i];

        jfloatArray row = env->NewFloatArray(6);
        float data[6] = {
            (float)obj.label,
            obj.prob,
            obj.rect.x,
            obj.rect.y,
            obj.rect.width,
            obj.rect.height
        };
        env->SetFloatArrayRegion(row, 0, 6, data);
        env->SetObjectArrayElement(result, i, row);
        env->DeleteLocalRef(row);
    }

    return result;
}

// ============================================================================
// JNI: nativeInit
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeInit(
    JNIEnv* env, jobject thiz)
{
    LOGD("nativeInit");
    return JNI_TRUE;
}

// ============================================================================
// JNI: nativeLoadModel
// Params: AssetManager, paramName, binName, numClass, sizeId, cpuGpu
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeLoadModel(
    JNIEnv* env, jobject thiz,
    jobject assetManager,
    jstring paramName, jstring binName,
    jint numClass, jint sizeId, jint cpuGpu)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr)
    {
        LOGE("nativeLoadModel: AAssetManager is null");
        return JNI_FALSE;
    }

    const char* paramStr = env->GetStringUTFChars(paramName, nullptr);
    const char* binStr = env->GetStringUTFChars(binName, nullptr);

    LOGD("nativeLoadModel: param=%s bin=%s numClass=%d sizeId=%d cpuGpu=%d",
         paramStr, binStr, (int)numClass, (int)sizeId, (int)cpuGpu);

    // Size options
    const int sizetypes[5] = { 320, 400, 480, 560, 640 };
    int target_size = 320;
    if (sizeId >= 0 && sizeId <= 4)
        target_size = sizetypes[sizeId];

    bool use_fp16 = true;
    bool use_gpu = (cpuGpu == 1);
    bool use_turnip = (cpuGpu == 2);

    {
        ncnn::MutexLockGuard g(g_lock);

        // Cleanup previous model
        delete g_picodet;
        g_picodet = nullptr;

        // Recreate GPU instance if needed
        ncnn::destroy_gpu_instance();
        if (use_turnip)
        {
            ncnn::create_gpu_instance("libvulkan_freedreno.so");
        }
        else if (use_gpu)
        {
            ncnn::create_gpu_instance();
        }

        g_picodet = new PicoDet;

        int ret = g_picodet->load(mgr, paramStr, binStr,
                                   (int)numClass, use_fp16, use_gpu || use_turnip);

        if (ret != 0)
        {
            LOGE("nativeLoadModel: load failed ret=%d", ret);
            delete g_picodet;
            g_picodet = nullptr;

            env->ReleaseStringUTFChars(paramName, paramStr);
            env->ReleaseStringUTFChars(binName, binStr);
            return JNI_FALSE;
        }

        g_picodet->set_target_size(target_size);
    }

    env->ReleaseStringUTFChars(paramName, paramStr);
    env->ReleaseStringUTFChars(binName, binStr);

    LOGD("nativeLoadModel: success, target_size=%d", target_size);
    return JNI_TRUE;
}

// ============================================================================
// JNI: nativeLoadModelFromFile
// Load model from absolute file paths (e.g., user-picked files)
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeLoadModelFromFile(
    JNIEnv* env, jobject thiz,
    jstring paramPath, jstring binPath,
    jint numClass, jint sizeId, jint cpuGpu)
{
    const char* paramStr = env->GetStringUTFChars(paramPath, nullptr);
    const char* binStr = env->GetStringUTFChars(binPath, nullptr);

    LOGD("nativeLoadModelFromFile: param=%s bin=%s numClass=%d sizeId=%d cpuGpu=%d",
         paramStr, binStr, (int)numClass, (int)sizeId, (int)cpuGpu);

    const int sizetypes[5] = { 320, 400, 480, 560, 640 };
    int target_size = 320;
    if (sizeId >= 0 && sizeId <= 4)
        target_size = sizetypes[sizeId];

    bool use_fp16 = true;
    bool use_gpu = (cpuGpu == 1);
    bool use_turnip = (cpuGpu == 2);

    {
        ncnn::MutexLockGuard g(g_lock);

        delete g_picodet;
        g_picodet = nullptr;

        ncnn::destroy_gpu_instance();
        if (use_turnip)
        {
            ncnn::create_gpu_instance("libvulkan_freedreno.so");
        }
        else if (use_gpu)
        {
            ncnn::create_gpu_instance();
        }

        g_picodet = new PicoDet;

        int ret = g_picodet->load(paramStr, binStr,
                                   (int)numClass, use_fp16, use_gpu || use_turnip);

        if (ret != 0)
        {
            LOGE("nativeLoadModelFromFile: load failed ret=%d", ret);
            delete g_picodet;
            g_picodet = nullptr;

            env->ReleaseStringUTFChars(paramPath, paramStr);
            env->ReleaseStringUTFChars(binPath, binStr);
            return JNI_FALSE;
        }

        g_picodet->set_target_size(target_size);
    }

    env->ReleaseStringUTFChars(paramPath, paramStr);
    env->ReleaseStringUTFChars(binPath, binStr);

    LOGD("nativeLoadModelFromFile: success, target_size=%d", target_size);
    return JNI_TRUE;
}

// ============================================================================
// JNI: nativeDetectFromPath
// Load image from file path, run detection
// ============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeDetectFromPath(
    JNIEnv* env, jobject thiz,
    jstring imagePath)
{
    const char* pathStr = env->GetStringUTFChars(imagePath, nullptr);

    cv::Mat bgr = cv::imread(pathStr, cv::IMREAD_COLOR);
    env->ReleaseStringUTFChars(imagePath, pathStr);

    if (bgr.empty())
    {
        LOGE("nativeDetectFromPath: failed to read image");
        // Return empty array
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    std::vector<DetObject> objects;
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_picodet)
        {
            g_picodet->detect(rgb, objects);
        }
        else
        {
            LOGE("nativeDetectFromPath: model not loaded");
        }
    }

    return detections_to_java(env, objects);
}

// ============================================================================
// JNI: nativeDetect (from Android Bitmap)
// ============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeDetect(
    JNIEnv* env, jobject thiz,
    jobject bitmap)
{
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS)
    {
        LOGE("nativeDetect: AndroidBitmap_getInfo failed");
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        LOGE("nativeDetect: unsupported bitmap format %d (need RGBA_8888)", info.format);
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS)
    {
        LOGE("nativeDetect: AndroidBitmap_lockPixels failed");
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<DetObject> objects;
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_picodet)
        {
            g_picodet->detect(rgb, objects);
        }
        else
        {
            LOGE("nativeDetect: model not loaded");
        }
    }

    return detections_to_java(env, objects);
}

// ============================================================================
// JNI: nativeDetectAndDraw (from Android Bitmap)
// Runs detection, draws bbox directly onto the bitmap, returns detection array.
// The bitmap is MODIFIED IN PLACE with drawn bounding boxes.
// ============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeDetectAndDraw(
    JNIEnv* env, jobject thiz,
    jobject bitmap)
{
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS)
    {
        LOGE("nativeDetectAndDraw: AndroidBitmap_getInfo failed");
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        LOGE("nativeDetectAndDraw: unsupported format %d", info.format);
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS)
    {
        LOGE("nativeDetectAndDraw: lockPixels failed");
        jclass floatArrayClass = env->FindClass("[F");
        return env->NewObjectArray(0, floatArrayClass, nullptr);
    }

    // Work with the pixel buffer directly
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

    std::vector<DetObject> objects;
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_picodet)
        {
            g_picodet->detect(rgb, objects);
            // Draw bbox onto rgb
            g_picodet->draw_detections(rgb, objects);
        }
        else
        {
            LOGE("nativeDetectAndDraw: model not loaded");
        }
    }

    // Convert back to RGBA and write into bitmap pixel buffer
    cv::cvtColor(rgb, rgba, cv::COLOR_RGB2RGBA);

    AndroidBitmap_unlockPixels(env, bitmap);

    return detections_to_java(env, objects);
}

// ============================================================================
// JNI: nativeDetectFromBytes (from raw pixel bytes, e.g., camera frame)
// Expects RGBA format byte array
// ============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeDetectFromBytes(
    JNIEnv* env, jobject thiz,
    jbyteArray data, jint width, jint height)
{
    jsize len = env->GetArrayLength(data);
    jbyte* bytes = env->GetByteArrayElements(data, nullptr);

    cv::Mat rgba(height, width, CV_8UC4, (void*)bytes);
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);

    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);

    std::vector<DetObject> objects;
    {
        ncnn::MutexLockGuard g(g_lock);
        if (g_picodet)
        {
            g_picodet->detect(rgb, objects);
        }
        else
        {
            LOGE("nativeDetectFromBytes: model not loaded");
        }
    }

    return detections_to_java(env, objects);
}

// ============================================================================
// JNI: nativeSetThreshold
// ============================================================================

JNIEXPORT void JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeSetThreshold(
    JNIEnv* env, jobject thiz,
    jfloat probThreshold, jfloat nmsThreshold)
{
    ncnn::MutexLockGuard g(g_lock);
    if (g_picodet)
    {
        g_picodet->set_prob_threshold(probThreshold);
        g_picodet->set_nms_threshold(nmsThreshold);
        LOGD("nativeSetThreshold: prob=%.3f nms=%.3f", probThreshold, nmsThreshold);
    }
}

// ============================================================================
// JNI: nativeGetNumClass
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeGetNumClass(
    JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(g_lock);
    if (g_picodet)
        return g_picodet->get_num_class();
    return 0;
}

// ============================================================================
// JNI: nativeSetAntiSpoof
// ============================================================================

JNIEXPORT void JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeSetAntiSpoof(
    JNIEnv* env, jobject thiz, jboolean enabled)
{
    ncnn::MutexLockGuard g(g_lock);
    if (g_picodet)
    {
        g_picodet->set_anti_spoof(enabled);
        LOGD("nativeSetAntiSpoof: %s", enabled ? "ON" : "OFF");
    }
}

// ============================================================================
// JNI: nativeSetLabels
// ============================================================================

JNIEXPORT void JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeSetLabels(
    JNIEnv* env, jobject thiz, jobjectArray labels)
{
    ncnn::MutexLockGuard g(g_lock);
    if (!g_picodet) return;

    std::vector<std::string> labelVec;
    if (labels != nullptr)
    {
        int len = env->GetArrayLength(labels);
        for (int i = 0; i < len; i++)
        {
            jstring jstr = (jstring)env->GetObjectArrayElement(labels, i);
            const char* str = env->GetStringUTFChars(jstr, nullptr);
            labelVec.push_back(str);
            env->ReleaseStringUTFChars(jstr, str);
            env->DeleteLocalRef(jstr);
        }
    }
    g_picodet->set_labels(labelVec);
    LOGD("nativeSetLabels: %d labels set", (int)labelVec.size());
}

// ============================================================================
// JNI: nativeIsAntiSpoofEnabled
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeIsAntiSpoofEnabled(
    JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(g_lock);
    if (g_picodet)
        return g_picodet->get_anti_spoof() ? JNI_TRUE : JNI_FALSE;
    return JNI_FALSE;
}

// ============================================================================
// JNI: nativeHasGpu
// Returns 1 if Vulkan GPU is available, 0 otherwise.
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeHasGpu(
    JNIEnv* env, jobject thiz)
{
#if NCNN_VULKAN
    return ncnn::get_gpu_count() > 0 ? 1 : 0;
#else
    return 0;
#endif
}

// ============================================================================
// JNI: nativeGetGpuCount
// Returns number of Vulkan GPU devices available.
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeGetGpuCount(
    JNIEnv* env, jobject thiz)
{
#if NCNN_VULKAN
    return ncnn::get_gpu_count();
#else
    return 0;
#endif
}

// ============================================================================
// JNI: nativeDispose
// ============================================================================

JNIEXPORT void JNICALL
Java_com_iweka_paddle_1detection_PaddleDetectionPlugin_nativeDispose(
    JNIEnv* env, jobject thiz)
{
    LOGD("nativeDispose");

    ncnn::MutexLockGuard g(g_lock);
    delete g_picodet;
    g_picodet = nullptr;
}

} // extern "C"
