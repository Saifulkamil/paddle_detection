# Changelog

## 0.0.2

### New Features
- 🔀 **Multi-Model Support**: Auto-detects model format at load time
  - Format A: Models with baked-in post-processing (finetune/custom, 2 inputs)
  - Format C: Models without post-processing (COCO pretrained, 1 input, recommended)
- 🧠 **FPN Decode**: Full DFL softmax + expectation decode for no-postprocess models
- 🏷️ **Dynamic Labels from File**: Load class names from `labels.txt` (asset or device file)
  - `loadLabelsFromAsset('labels.txt')` — from Android assets
  - `loadLabelsFromFile('/path/to/labels.txt')` — from device storage
  - `setLabels(['class0', 'class1', ...])` — set programmatically
  - No hardcoded label names in C++ — all driven by file
- 🎛️ **Runtime Model Loading**: Pick .param, .bin, and labels.txt separately from device
- ⚡ **GPU Toggle**: Switch CPU/GPU (Vulkan) from settings with auto-detection
- 🛡️ **Anti-Spoof**: FFT moiré pattern detection to reject screen/monitor images
- 📱 **Orientation Aware**: Camera preview rotates with device physical orientation
- 📷 **Capture Guard**: Photo capture only enabled when objects are detected
- ▶️ **Start/Stop Camera**: Manual control over detection pipeline

### Improvements
- Optimized pipeline: reduced from 4 bitmap copies to 1 per frame
- On-demand capture allocation (no per-frame overhead)
- Auto num_class detection from model blob shape via dummy inference at load time
- Handles transposed score/box tensors automatically (min/max dimension heuristic)
- Separate .param, .bin, and labels.txt file picker buttons
- GPU availability check before enabling toggle
- `Resize` → `Interp` layer replacement documented for model conversion

### Bug Fixes
- Fixed bbox position mismatch in camera (draw in native C++)
- Fixed threshold not applying to all classes equally
- Fixed capture crash (recycled bitmap race condition)
- Fixed camera black screen (runtime permission)
- Fixed landscape detection (OrientationEventListener + targetRotation)
- Fixed preview flip when device rotated
- Fixed GPU switch not responsive in settings sheet
- Fixed model load failure due to mismatched .bin filename
- Fixed Resize layer not supported (replaced with Interp in .param)
- Fixed num_class showing 1600 instead of actual (min dimension heuristic)
- Fixed labels showing COCO names for custom models (now file-driven)
- Fixed num_class = 2 for 5-class model (dummy inference at load)
- Fixed scores/boxes tensor transpose for different model exports

### Model Support
- PP-PicoDet S/M/L (320×320)
- COCO 80-class pretrained models (no postprocess)
- Custom finetune models (any number of classes, with postprocess)
- ONNX → NCNN via PNNX conversion documented
- Labels loaded from external text file (not hardcoded)

## 0.0.1

- Initial project setup
- Basic NCNN inference pipeline
- JNI bridge + Kotlin plugin
- Dart API + method channel
