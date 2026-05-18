package com.iweka.paddle_detection

import android.app.Activity
import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.hardware.SensorManager
import android.util.Size
import android.view.OrientationEventListener
import android.view.Surface
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.*
import io.flutter.view.TextureRegistry
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class PaddleDetectionPlugin : FlutterPlugin, MethodChannel.MethodCallHandler,
    EventChannel.StreamHandler, ActivityAware {

    private lateinit var channel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private var assetManager: AssetManager? = null
    private var context: Context? = null
    private var activity: Activity? = null
    private var textureRegistry: TextureRegistry? = null
    private var flutterPluginBinding: FlutterPlugin.FlutterPluginBinding? = null

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraControl: CameraControl? = null
    private var textureEntry: TextureRegistry.SurfaceTextureEntry? = null
    private var cameraExecutor: ExecutorService? = null
    private var eventSink: EventChannel.EventSink? = null
    private val isProcessing = AtomicBoolean(false)
    private var previewWidth = 0
    private var previewHeight = 0
    private var flashEnabled = false
    private var useFrontCamera = false
    private var orientationListener: OrientationEventListener? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var currentDeviceRotation = 0

    // Last frame for capture
    @Volatile private var lastAnnotatedBitmap: Bitmap? = null
    @Volatile private var lastDetections: Array<FloatArray>? = null
    @Volatile private var captureRequested = false
    @Volatile private var capturedCleanBitmap: Bitmap? = null
    @Volatile private var capturedAnnotatedBitmap: Bitmap? = null
    @Volatile private var capturedDetections: Array<FloatArray>? = null
    private var pendingCaptureResult: MethodChannel.Result? = null
    private var pendingCaptureFolder: String? = null
    private var pendingCapturePrefix: String? = null

    companion object {
        init {
            System.loadLibrary("paddle_detection_jni")
        }
    }

    // Native methods
    private external fun nativeInit(): Boolean
    private external fun nativeLoadModel(mgr: AssetManager, paramName: String, binName: String, numClass: Int, sizeId: Int, cpuGpu: Int): Boolean
    private external fun nativeLoadModelFromFile(paramPath: String, binPath: String, numClass: Int, sizeId: Int, cpuGpu: Int): Boolean
    private external fun nativeDetect(bitmap: Bitmap): Array<FloatArray>?
    private external fun nativeDetectAndDraw(bitmap: Bitmap): Array<FloatArray>?
    private external fun nativeDetectFromPath(imagePath: String): Array<FloatArray>?
    private external fun nativeDetectFromBytes(data: ByteArray, width: Int, height: Int): Array<FloatArray>?
    private external fun nativeSetThreshold(probThreshold: Float, nmsThreshold: Float)
    private external fun nativeGetNumClass(): Int
    private external fun nativeHasGpu(): Int
    private external fun nativeGetGpuCount(): Int
    private external fun nativeSetAntiSpoof(enabled: Boolean)
    private external fun nativeIsAntiSpoofEnabled(): Boolean
    private external fun nativeSetLabels(labels: Array<String>?)
    private external fun nativeDispose()

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        flutterPluginBinding = binding
        channel = MethodChannel(binding.binaryMessenger, "paddle_detection")
        channel.setMethodCallHandler(this)
        eventChannel = EventChannel(binding.binaryMessenger, "paddle_detection/detections")
        eventChannel.setStreamHandler(this)
        context = binding.applicationContext
        assetManager = binding.applicationContext.assets
        textureRegistry = binding.textureRegistry
        nativeInit()
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        stopCamera()
        flutterPluginBinding = null
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) { activity = binding.activity }
    override fun onDetachedFromActivityForConfigChanges() { activity = null }
    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) { activity = binding.activity }
    override fun onDetachedFromActivity() { activity = null }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { eventSink = events }
    override fun onCancel(arguments: Any?) { eventSink = null }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
            "loadModel" -> handleLoadModel(call, result)
            "loadModelFromFile" -> handleLoadModelFromFile(call, result)
            "detect" -> handleDetect(call, result)
            "detectFromBytes" -> handleDetectFromBytes(call, result)
            "setThreshold" -> handleSetThreshold(call, result)
            "getNumClass" -> result.success(nativeGetNumClass())
            "hasGpu" -> result.success(nativeHasGpu() > 0)
            "getGpuCount" -> result.success(nativeGetGpuCount())
            "setAntiSpoof" -> { nativeSetAntiSpoof(call.argument<Boolean>("enabled") ?: false); result.success(true) }
            "getAntiSpoof" -> result.success(nativeIsAntiSpoofEnabled())
            "setLabels" -> {
                val labels = call.argument<List<String>>("labels")
                nativeSetLabels(labels?.toTypedArray())
                result.success(true)
            }
            "loadLabelsFromAsset" -> {
                val fileName = call.argument<String>("fileName") ?: return result.error("INVALID_ARGS", "fileName required", null)
                try {
                    val mgr = assetManager ?: return result.error("NO_CONTEXT", "AssetManager null", null)
                    val labels = mgr.open(fileName).bufferedReader().readLines().filter { it.isNotBlank() }
                    nativeSetLabels(labels.toTypedArray())
                    result.success(mapOf("labels" to labels, "count" to labels.size))
                } catch (e: Exception) {
                    result.error("LABEL_ERROR", "Failed to load labels: ${e.message}", null)
                }
            }
            "loadLabelsFromFile" -> {
                val path = call.argument<String>("path") ?: return result.error("INVALID_ARGS", "path required", null)
                try {
                    val labels = java.io.File(path).readLines().filter { it.isNotBlank() }
                    nativeSetLabels(labels.toTypedArray())
                    result.success(mapOf("labels" to labels, "count" to labels.size))
                } catch (e: Exception) {
                    result.error("LABEL_ERROR", "Failed to load labels: ${e.message}", null)
                }
            }
            "startCamera" -> handleStartCamera(call, result)
            "stopCamera" -> { stopCamera(); result.success(true) }
            "toggleFlash" -> handleToggleFlash(call, result)
            "switchCamera" -> handleSwitchCamera(call, result)
            "capturePhoto" -> handleCapturePhoto(call, result)
            "dispose" -> { stopCamera(); nativeDispose(); result.success(true) }
            else -> result.notImplemented()
        }
    }

    // =========================================================================
    // Model loading
    // =========================================================================

    private fun handleLoadModel(call: MethodCall, result: MethodChannel.Result) {
        val paramName = call.argument<String>("paramName") ?: return result.error("INVALID_ARGS", "paramName required", null)
        val binName = call.argument<String>("binName") ?: return result.error("INVALID_ARGS", "binName required", null)
        val numClass = call.argument<Int>("numClass") ?: 2
        val sizeId = call.argument<Int>("sizeId") ?: 0
        val cpuGpu = call.argument<Int>("cpuGpu") ?: 0
        val mgr = assetManager ?: return result.error("NO_CONTEXT", "AssetManager null", null)
        val ok = nativeLoadModel(mgr, paramName, binName, numClass, sizeId, cpuGpu)
        if (ok) {
            val actualClasses = nativeGetNumClass()
            result.success(mapOf("success" to true, "numClass" to actualClasses))
        } else {
            result.success(mapOf("success" to false, "numClass" to 0))
        }
    }

    private fun handleLoadModelFromFile(call: MethodCall, result: MethodChannel.Result) {
        val paramPath = call.argument<String>("paramPath") ?: return result.error("INVALID_ARGS", "paramPath required", null)
        val binPath = call.argument<String>("binPath") ?: return result.error("INVALID_ARGS", "binPath required", null)
        val numClass = call.argument<Int>("numClass") ?: 2
        val sizeId = call.argument<Int>("sizeId") ?: 0
        val cpuGpu = call.argument<Int>("cpuGpu") ?: 0
        val ok = nativeLoadModelFromFile(paramPath, binPath, numClass, sizeId, cpuGpu)
        if (ok) {
            val actualClasses = nativeGetNumClass()
            result.success(mapOf("success" to true, "numClass" to actualClasses))
        } else {
            result.success(mapOf("success" to false, "numClass" to 0))
        }
    }

    // =========================================================================
    // Detection
    // =========================================================================

    private fun handleDetect(call: MethodCall, result: MethodChannel.Result) {
        val imagePath = call.argument<String>("imagePath") ?: return result.error("INVALID_ARGS", "imagePath required", null)
        result.success(convertDetections(nativeDetectFromPath(imagePath)))
    }

    private fun handleDetectFromBytes(call: MethodCall, result: MethodChannel.Result) {
        val data = call.argument<ByteArray>("data") ?: return result.error("INVALID_ARGS", "data required", null)
        val width = call.argument<Int>("width") ?: return result.error("INVALID_ARGS", "width required", null)
        val height = call.argument<Int>("height") ?: return result.error("INVALID_ARGS", "height required", null)
        result.success(convertDetections(nativeDetectFromBytes(data, width, height)))
    }

    private fun handleSetThreshold(call: MethodCall, result: MethodChannel.Result) {
        val prob = call.argument<Double>("probThreshold")?.toFloat() ?: 0.5f
        val nms = call.argument<Double>("nmsThreshold")?.toFloat() ?: 0.45f
        nativeSetThreshold(prob, nms)
        result.success(true)
    }

    // =========================================================================
    // Flash
    // =========================================================================

    private fun handleToggleFlash(call: MethodCall, result: MethodChannel.Result) {
        val enable = call.argument<Boolean>("enable") ?: !flashEnabled
        flashEnabled = enable
        cameraControl?.enableTorch(enable)
        result.success(flashEnabled)
    }

    // =========================================================================
    // Switch camera
    // =========================================================================

    private fun handleSwitchCamera(call: MethodCall, result: MethodChannel.Result) {
        useFrontCamera = !useFrontCamera
        // Restart camera with new selector
        val ctx = context
        if (ctx != null && cameraProvider != null && textureEntry != null) {
            // Rebind with new camera
            try {
                cameraProvider?.unbindAll()

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                analysis.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImageProxy(imageProxy)
                }

                val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                val lifecycleOwner = activity as? LifecycleOwner ?: ProcessLifecycleOwner.get()
                val camera = cameraProvider?.bindToLifecycle(lifecycleOwner, selector, analysis)
                cameraControl = camera?.cameraControl
                flashEnabled = false

                result.success(useFrontCamera)
            } catch (e: Exception) {
                result.error("SWITCH_ERROR", e.message, null)
            }
        } else {
            result.success(useFrontCamera)
        }
    }

    // =========================================================================
    // Capture photo — saves clean + annotated images to specified folder
    // =========================================================================

    private fun handleCapturePhoto(call: MethodCall, result: MethodChannel.Result) {
        val folder = call.argument<String>("folder") ?: return result.error("INVALID_ARGS", "folder required", null)
        val prefix = call.argument<String>("prefix") ?: "capture"

        // Request capture on next frame (so we get both clean + annotated)
        pendingCaptureResult = result
        pendingCaptureFolder = folder
        pendingCapturePrefix = prefix
        captureRequested = true
    }

    // =========================================================================
    // Camera
    // =========================================================================

    private fun handleStartCamera(call: MethodCall, result: MethodChannel.Result) {
        val ctx = context ?: return result.error("NO_CONTEXT", "Context null", null)

        stopCamera()

        cameraExecutor = Executors.newSingleThreadExecutor()
        textureEntry = textureRegistry?.createSurfaceTexture()
        if (textureEntry == null) return result.error("TEXTURE_ERROR", "Cannot create texture", null)

        val textureId = textureEntry!!.id()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                analysis.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImageProxy(imageProxy)
                }

                val selector = if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                cameraProvider?.unbindAll()

                val lifecycleOwner = activity as? LifecycleOwner ?: ProcessLifecycleOwner.get()
                val camera = cameraProvider?.bindToLifecycle(lifecycleOwner, selector, analysis)
                cameraControl = camera?.cameraControl
                imageAnalysis = analysis

                // Listen for device orientation changes to update analysis rotation
                orientationListener = object : OrientationEventListener(ctx, SensorManager.SENSOR_DELAY_NORMAL) {
                    override fun onOrientationChanged(orientation: Int) {
                        if (orientation == ORIENTATION_UNKNOWN) return
                        val rotation = when {
                            orientation >= 315 || orientation < 45 -> Surface.ROTATION_0
                            orientation in 45..134 -> Surface.ROTATION_270
                            orientation in 135..224 -> Surface.ROTATION_180
                            else -> Surface.ROTATION_90
                        }
                        imageAnalysis?.targetRotation = rotation
                        // Store current device rotation for Flutter
                        currentDeviceRotation = when (rotation) {
                            Surface.ROTATION_0 -> 0
                            Surface.ROTATION_90 -> 90
                            Surface.ROTATION_180 -> 180
                            Surface.ROTATION_270 -> 270
                            else -> 0
                        }
                    }
                }
                orientationListener?.enable()

                previewWidth = 480
                previewHeight = 640

                result.success(mapOf(
                    "textureId" to textureId,
                    "previewWidth" to previewWidth,
                    "previewHeight" to previewHeight
                ))
            } catch (e: Exception) {
                result.error("CAMERA_ERROR", e.message, null)
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            val mutableBitmap = if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap.copy(Bitmap.Config.ARGB_8888, true)
            }

            val imgW = mutableBitmap.width
            val imgH = mutableBitmap.height

            // If capture requested, save clean copy BEFORE drawing
            val doCapture = captureRequested
            var cleanForCapture: Bitmap? = null
            if (doCapture) {
                cleanForCapture = mutableBitmap.copy(Bitmap.Config.ARGB_8888, false)
                captureRequested = false
            }

            val t0 = System.currentTimeMillis()
            val detections = nativeDetectAndDraw(mutableBitmap)
            val inferenceMs = System.currentTimeMillis() - t0

            // Store for reference
            lastAnnotatedBitmap?.recycle()
            lastAnnotatedBitmap = mutableBitmap
            lastDetections = detections

            // Render to texture
            val surfaceTexture = textureEntry?.surfaceTexture()
            if (surfaceTexture != null) {
                surfaceTexture.setDefaultBufferSize(imgW, imgH)
                val surface = Surface(surfaceTexture)
                val canvas = surface.lockCanvas(null)
                canvas.drawBitmap(mutableBitmap, 0f, 0f, null)
                surface.unlockCanvasAndPost(canvas)
                surface.release()
            }

            // Handle capture save (only on capture frame)
            if (doCapture && cleanForCapture != null) {
                val folder = pendingCaptureFolder ?: ""
                val prefix = pendingCapturePrefix ?: "capture"
                val result = pendingCaptureResult
                val annotatedForCapture = mutableBitmap.copy(Bitmap.Config.ARGB_8888, false)
                pendingCaptureResult = null
                pendingCaptureFolder = null
                pendingCapturePrefix = null

                // Save in background
                cameraExecutor?.execute {
                    try {
                        val dir = File(folder)
                        if (!dir.exists()) dir.mkdirs()
                        val ts = System.currentTimeMillis()
                        val cleanPath = File(dir, "${prefix}_${ts}.jpg").absolutePath
                        val annotatedPath = File(dir, "${prefix}_${ts}_bbox.jpg").absolutePath

                        FileOutputStream(cleanPath).use { cleanForCapture.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                        FileOutputStream(annotatedPath).use { annotatedForCapture.compress(Bitmap.CompressFormat.JPEG, 95, it) }

                        cleanForCapture.recycle()
                        annotatedForCapture.recycle()

                        val detList = convertDetections(detections)
                        val response = mapOf("cleanPath" to cleanPath, "annotatedPath" to annotatedPath, "detections" to detList)
                        activity?.runOnUiThread { result?.success(response) }
                    } catch (e: Exception) {
                        cleanForCapture.recycle()
                        annotatedForCapture.recycle()
                        activity?.runOnUiThread { result?.error("CAPTURE_ERROR", e.message, null) }
                    }
                }
            }

            // Stream to Flutter
            val results = convertDetections(detections)
            val payload = mapOf("detections" to results, "imageWidth" to imgW, "imageHeight" to imgH, "inferenceMs" to inferenceMs, "deviceRotation" to currentDeviceRotation)
            activity?.runOnUiThread { eventSink?.success(payload) }
        } catch (_: Exception) {
        } finally {
            imageProxy.close()
            isProcessing.set(false)
        }
    }

    private fun stopCamera() {
        orientationListener?.disable()
        orientationListener = null
        imageAnalysis = null
        cameraProvider?.unbindAll()
        cameraProvider = null
        cameraControl = null
        textureEntry?.release()
        textureEntry = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        lastAnnotatedBitmap?.recycle()
        lastAnnotatedBitmap = null
        lastDetections = null
    }

    // =========================================================================
    private fun convertDetections(detections: Array<FloatArray>?): List<Map<String, Any>> {
        return detections?.map { det ->
            mapOf("label" to det[0].toInt(), "prob" to det[1].toDouble(),
                "x" to det[2].toDouble(), "y" to det[3].toDouble(),
                "width" to det[4].toDouble(), "height" to det[5].toDouble())
        } ?: emptyList()
    }
}
