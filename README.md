# Fast Paddle Detection 🎯

[![pub package](https://img.shields.io/pub/v/fast_paddle_detection.svg)](https://pub.dev/packages/fast_paddle_detection)
[![License](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](LICENSE)

A high-performance Flutter plugin for **offline object detection** on Android using **PP-PicoDet** (PaddleDetection) and **NCNN**.

This plugin supports real-time detection directly from the camera feed, photo capture with bounding box visualization, and on-device inference. It processes everything locally using C++ (NCNN), meaning it is **extremely fast** and requires **zero internet connection**.

## ✨ Key Features

- ⚡ **Real-time Detection**: Detect objects instantly from the live camera preview.
- 🎯 **Native C++ Bbox Drawing**: Bounding boxes drawn directly on the frame in C++ — zero coordinate mapping issues.
- 🔀 **Multi-Model Support**: Auto-detects model format (with/without post-processing). Load COCO 80-class or custom finetune models.
- 📸 **Photo Capture**: Save both clean and annotated (with bbox) images on detection.
- 🔦 **Flash Control**: Toggle camera flash/torch on/off.
- 🔄 **Switch Camera**: Toggle between front and back camera.
- 📱 **Orientation Aware**: Camera preview rotates with device physical orientation (app stays portrait).
- 🎛️ **Runtime Model Loading**: Pick and load custom models from device storage at runtime.
- 🏷️ **Dynamic Labels**: Load class names from `labels.txt` (asset or device file). No hardcoded labels.
- 🧠 **Auto Class Detection**: Number of classes detected automatically from model structure.
- ⚡ **GPU Acceleration**: Optional Vulkan GPU inference with auto-detection of GPU availability.
- 📴 **100% Offline**: Uses local NCNN models. No API calls or cloud dependencies.
- 🔋 **Optimized Performance**: Minimal bitmap copies per frame, on-demand capture allocation.
- 🛡️ **Anti-Spoof Detection**: Detect and reject images from screens/monitors using moiré pattern analysis (FFT-based).

---

## 🎥 Demo

*(Replace placeholder links with your uploaded media)*


 <table align="center">
  <tr>
    <td align="center"><b>Real-Time Detection</b></td>
    <td align="center"><b>Photo Capture Result</b></td>
  </tr>
  <tr>
    <td align="center">
      <img 
        src="https://github.com/user-attachments/assets/c0c2a589-fd61-400c-9d30-6fae2b47826f"
        alt="Real-Time Detection"
        width="250"
      >
    </td>
    <td align="center">
      <img 
        src="https://github.com/user-attachments/assets/40f46ae4-3f30-4df2-b7f6-c0ecd9b572d1"
        alt="Photo Capture Result"
        width="250"
      >
    </td>
  </tr>
</table>

## 📱 Platform Support

| Platform    | Support | Note                                             |
| :---------- | :-----: | :----------------------------------------------- |
| **Android** |   ✅    | Fully supported (Requires Android 7.0 / API 24+) |
| **iOS**     |   ❌    | Not supported                                    |
| **Web**     |   ❌    | Not supported                                    |
| **Desktop** |   ❌    | Not supported                                    |

---

## ⚙️ Android Setup (Required)

### 1. Minimum SDK Version

Ensure your `android/app/build.gradle.kts` has `minSdk` of at least `24`:

```kotlin
android {
    defaultConfig {
        minSdk = 24
    }
}
```

### 2. NDK Version

Set NDK version to match the plugin:

```kotlin
android {
    ndkVersion = "29.0.14206865"
}
```

### 3. Reduce APK Size (Crucial) 🚨

NCNN libraries are large. Filter architectures to only physical ARM devices:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
}
```

### 4. Camera Permission

Add to your `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-feature android:name="android.hardware.camera" android:required="false"/>
```

Request permission at runtime using `permission_handler` package.

### 5. Model Files

Place your NCNN model files in `android/app/src/main/assets/`:

```
android/app/src/main/assets/
├── model.ncnn.param
└── model.ncnn.bin
```

---

## 🚀 Usage

### 1. Import

```dart
import 'package:fast_paddle_detection/paddle_detection.dart';
```

### 2. Load Model

```dart
final detector = PaddleDetection();

// Load from assets (default CPU)
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
);
print('Classes: ${info.numClass}'); // auto-detected from model

// Or load with GPU
final hasGpu = await detector.hasGpu();
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
  cpuGpu: hasGpu ? 1 : 0, // 0=CPU, 1=GPU(Vulkan), 2=GPU(Turnip)
);

// Or load from file path (user-picked)
final info = await detector.loadModelFromFile(
  paramPath: '/path/to/model.ncnn.param',
  binPath: '/path/to/model.ncnn.bin',
);
```

### 3. Set Threshold

```dart
// Single value controls both prob and NMS threshold
await detector.setThreshold(threshold: 0.8);
```

### 4. Real-time Camera Detection

```dart
// Start camera (returns texture info for display)
final cam = await detector.startCamera();

// Display in Flutter
Texture(textureId: cam.textureId)

// Listen to detection stream
detector.detectionStream.listen((event) {
  print('${event.detections.length} objects, ${event.inferenceMs}ms');
  print('Device rotation: ${event.deviceRotation}°');
});

// Toggle flash
await detector.toggleFlash(enable: true);

// Switch front/back camera
await detector.switchCamera();

// Stop camera
await detector.stopCamera();
```

### 5. Capture Photo

Capture saves both a clean image and an annotated image (with bbox drawn):

```dart
final result = await detector.capturePhoto(
  folder: '/storage/emulated/0/DCIM/MyApp',
  prefix: 'detection',
);

print(result.cleanPath);     // /path/detection_1234567890.jpg
print(result.annotatedPath); // /path/detection_1234567890_bbox.jpg
print(result.detections);    // List<DetectionResult>
```

> **Note**: Capture button is only active when objects are detected.

### 6. Detect from Image File (Gallery)

```dart
final results = await detector.detect('/path/to/image.jpg');
for (final det in results) {
  print('${det.label} ${det.probability} ${det.x},${det.y},${det.width},${det.height}');
}
```

### 7. GPU Detection

```dart
final hasGpu = await detector.hasGpu();     // true if Vulkan available
final gpuCount = await detector.getGpuCount(); // number of GPU devices
```

### 8. Anti-Spoof (Screen Detection)

When enabled, the plugin checks if the camera is pointed at a screen/monitor. If detected, object detection is skipped (returns empty results).

```dart
// Enable anti-spoof
await detector.setAntiSpoof(enabled: true);

// Disable anti-spoof
await detector.setAntiSpoof(enabled: false);

// Check current state
final isOn = await detector.getAntiSpoof();
```

**How it works:**
- Converts frame to grayscale → FFT frequency analysis
- Screens emit moiré patterns (periodic high-frequency artifacts from pixel grid)
- If moiré pattern detected → frame is from screen → skip detection
- Real-world objects pass through → normal detection runs

### 9. Cleanup

```dart
await detector.stopCamera();
await detector.dispose();
```

---

## 📝 API Reference

| Method | Description |
| :----- | :---------- |
| `loadModel(paramName, binName, cpuGpu)` | Load model from Android assets |
| `loadModelFromFile(paramPath, binPath, cpuGpu)` | Load model from absolute file paths |
| `setThreshold(threshold)` | Set detection threshold (0.0–1.0) |
| `getNumClass()` | Get number of classes from loaded model |
| `hasGpu()` | Check if Vulkan GPU is available |
| `getGpuCount()` | Get number of GPU devices |
| `setAntiSpoof(enabled)` | Enable/disable screen detection (anti-spoof) |
| `getAntiSpoof()` | Check if anti-spoof is enabled |
| `setLabels(labels)` | Set custom label names for the model |
| `loadLabelsFromAsset(fileName)` | Load label names from assets text file |
| `loadLabelsFromFile(path)` | Load label names from device storage text file |
| `detect(imagePath)` | Run detection on image file |
| `detectFromBytes(data, width, height)` | Run detection on raw RGBA bytes |
| `startCamera()` | Start native camera, returns texture info |
| `stopCamera()` | Stop native camera |
| `switchCamera()` | Toggle front/back camera |
| `toggleFlash(enable)` | Toggle camera flash |
| `capturePhoto(folder, prefix)` | Capture clean + annotated images |
| `detectionStream` | Stream of realtime detection events |
| `dispose()` | Release all native resources |

### Data Classes

```dart
class ModelInfo {
  final bool success;
  final int numClass; // auto-detected from model
}

class DetectionResult {
  final int label;
  final double probability;
  final double x, y, width, height; // absolute pixels
}

class CameraInfo {
  final int textureId;
  final int previewWidth, previewHeight;
}

class CameraDetectionEvent {
  final List<DetectionResult> detections;
  final int imageWidth, imageHeight;
  final int inferenceMs;
  final int deviceRotation; // 0, 90, 180, 270
}

class CaptureResult {
  final String cleanPath;     // image without bbox
  final String annotatedPath; // image with bbox
  final List<DetectionResult> detections;
}
```

---

## 🧠 Model Specification

This plugin supports **two model formats** automatically detected at load time:

### Format A: Model WITH Post-Processing (Finetune/Custom)

For models exported from PaddleDetection **with** baked-in NMS/TopK. The plugin extracts intermediate blobs before the unsupported post-processing layers.

- **2 inputs**: `in0` = metadata `[4]`, `in1` = pixels `[3,320,320]`
- **Intermediate blobs**: scores + boxes extracted before NMS
- **Stub layers**: `pnnx.Expression`, `NonMaxSuppression`, `TopK`, `Gather`, `F.embedding`, `Tensor.to`

### Format C: Model WITHOUT Post-Processing (Recommended ✅)

For models exported **without** post-processing. All layers are standard ncnn ops — no stubs needed. The plugin performs FPN decode + DFL + NMS in C++.

- **1 input**: `in0` = pixels `[3,320,320]`
- **8 outputs**: `out0-3` (cls per stride), `out4-7` (dis per stride)
- **Strides**: 8, 16, 32, 64
- **DFL reg_max**: 7 (32 bins = 4 × 8)

### Labels

- **2-class model** → labels: "LCK", "SCR" (or whatever is in your labels.txt)
- **80-class model** → labels: COCO names (person, bicycle, car, cup, etc.)
- **Custom model** → provide a `labels.txt` file (one name per line, order = class index)
- Auto-loaded from `labels.txt` in assets at startup
- Can be picked from device storage at runtime

**labels.txt format:**
```
person
bicycle
car
...
```
(Line 0 = class 0, line 1 = class 1, etc.)

---

## 🔧 Using Pretrained PicoDet Models

You can download pretrained PicoDet models from the official PaddleDetection repository:

👉 **[PaddleDetection PicoDet Model Zoo](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/README_en.md)**

### Recommended Models (320×320 input)

| Model | mAP | Size | Speed |
|-------|-----|------|-------|
| PicoDet-S 320 | 29.1 | 4.4MB | Fast |
| PicoDet-M 320 | 34.4 | 11.0MB | Medium |
| PicoDet-L 320 | 36.1 | 14.0MB | Slower |

### Step-by-Step: Download & Convert Model

#### 1. Download ONNX Model (without post-processing)

From PaddleDetection model zoo, download the **ONNX model without post-processing**:

```
picodet_s_320_coco_lcnet.onnx        (NOT the _postprocessed version)
picodet_m_320_coco_lcnet.onnx
picodet_l_320_coco_lcnet.onnx
```

> ⚠️ **Important**: Download the version **WITHOUT** `_postprocessed` in the filename. The plugin handles post-processing in C++.

#### 2. Convert ONNX → NCNN using PNNX

Install PNNX from [ncnn releases](https://github.com/Tencent/ncnn/releases), then:

```bash
pnnx picodet_s_320_coco_lcnet.onnx inputshape=[1,3,320,320]f32
```

This produces:
- `picodet_s_320_coco_lcnet.ncnn.param`
- `picodet_s_320_coco_lcnet.ncnn.bin`

#### 3. Fix `Resize` Layers (if present)

Open the `.ncnn.param` file and check if there are `Resize` layers. If yes, replace them with `Interp`:

**Find lines like:**
```
Resize    Resize_162    1 1 82 85
```

**Replace with:**
```
Interp    Resize_162    1 1 82 85 0=1 1=2.0 2=2.0 6=0
```

(Parameters: `0=1` nearest mode, `1=2.0` height scale, `2=2.0` width scale, `6=0` align corner)

#### 4. Place in Assets

Copy both files to your Flutter project:
```
android/app/src/main/assets/
├── model.ncnn.param    (renamed from picodet_s_320_coco_lcnet.ncnn.param)
└── model.ncnn.bin      (renamed from picodet_s_320_coco_lcnet.ncnn.bin)
```

#### 5. Load in Code

```dart
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
);
print('Loaded ${info.numClass} classes'); // 80 for COCO
```

### Converting Your Own Finetune Model

If you trained a custom PicoDet model with PaddleDetection:

#### Option A: Export WITHOUT post-processing (Recommended)

```bash
# Export from PaddleDetection
python tools/export_model.py \
  -c configs/picodet/your_config.yml \
  -o weights=output/best_model.pdparams \
     export.benchmark=True \
     export.post_process=False \
     export.nms=False \
  --output_dir=inference_model

# Convert to ONNX
paddle2onnx --model_dir inference_model/your_model \
  --model_filename model.pdmodel \
  --params_filename model.pdiparams \
  --save_file model.onnx \
  --opset_version 11

# (Optional) Simplify
python -m onnxsim model.onnx model_sim.onnx

# Convert to NCNN
pnnx model_sim.onnx inputshape=[1,3,320,320]f32
```

#### Option B: Export WITH post-processing

```bash
# Export with default post-processing
python tools/export_model.py \
  -c configs/picodet/your_config.yml \
  -o weights=output/best_model.pdparams \
  --output_dir=inference_model

# Convert to ONNX
paddle2onnx --model_dir inference_model/your_model \
  --model_filename model.pdmodel \
  --params_filename model.pdiparams \
  --save_file model.onnx \
  --opset_version 11

# Convert to NCNN (2 inputs for postprocessed model)
pnnx model.onnx inputshape=[1,2]f32,[1,3,320,320]f32
```

> Note: Option B models may need `Resize` → `Interp` replacement in the `.param` file.

### Loading Custom Model at Runtime

Users can also load models from device storage at runtime:

```dart
final info = await detector.loadModelFromFile(
  paramPath: '/storage/emulated/0/Download/my_model.ncnn.param',
  binPath: '/storage/emulated/0/Download/my_model.ncnn.bin',
);
```

Or use the built-in file picker in the example app (Settings → Pick .param → Pick .bin → Load Custom).

---

## 💡 Tips & Best Practices

1. **Memory Management**: Always call `stopCamera()` and `dispose()` in your widget's `dispose()` method.
2. **Threshold Tuning**: Start with `0.5` and increase to reduce false positives. For production, `0.7–0.9` is typical.
3. **Capture Guard**: The plugin only allows capture when objects are detected — no empty captures.
4. **Flash**: Only available on back camera. Automatically disabled when switching to front.
5. **Model Size**: Use lightweight PicoDet models (PP-PicoDet-S) for mobile. Larger models will be slower.
6. **GPU Mode**: Set `cpuGpu: 1` for Vulkan GPU acceleration on supported devices. CPU (`cpuGpu: 0`) is more stable across all devices. Not all devices are faster with GPU — test both.
7. **Debug FPS**: In debug builds, FPS and inference time are shown as an overlay badge.
8. **Orientation**: Camera preview automatically rotates with device physical orientation while app stays portrait.
9. **Num Classes**: The plugin auto-detects the number of classes from the model's blob shape — no need to specify manually.
10. **Anti-Spoof**: Enable `setAntiSpoof(enabled: true)` to prevent detection from screen/monitor images. Useful for validation scenarios where you need real-world captures only. Threshold may need tuning for your specific use case.

---

## 🏗️ Architecture

```
Flutter (Dart API)
    │
    ├── MethodChannel "paddle_detection"
    └── EventChannel "paddle_detection/detections"
          │
          ▼
Kotlin (CameraX + Plugin)
    │
    ├── CameraX ImageAnalysis → frame capture
    ├── OrientationEventListener → dynamic rotation
    ├── Rotate bitmap → JNI call
    ├── Render annotated frame → Flutter Texture
    └── Stream detections + inferenceMs + deviceRotation → Dart
          │
          ▼
C++ / JNI
    │
    ├── Auto-detect model format (A/C) at load time
    ├── NCNN inference (ncnn-20260113-android-vulkan)
    │     ├── Format A: stub layers + extract intermediate blobs
    │     └── Format C: standard FPN decode (DFL + softmax + NMS)
    ├── Anti-spoof (FFT moiré pattern detection)
    ├── Dynamic labels (COCO 80 or custom 2-class)
    ├── OpenCV bbox drawing (rectangle + putText)
    └── OpenCV Mobile 4.13.0 (core + imgproc + highgui)
```

---

## 📄 License

This project is licensed under [CC0 1.0 Universal](LICENSE) — public domain dedication.
