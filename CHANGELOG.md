# Changelog

## 0.0.3

### New Features
- 🏷️ **Dynamic Labels from File**: Load class names from `labels.txt`
  - `loadLabelsFromAsset('labels.txt')` — from Android assets
  - `loadLabelsFromFile('/path/to/labels.txt')` — pick from device storage
  - `setLabels(['class0', 'class1', ...])` — set programmatically
  - No hardcoded label names — all driven by external file
  - Pick labels button in settings UI

### Bug Fixes
- Fixed num_class always showing 2 for multi-class models (now uses dummy inference at load time)
- Fixed scores/boxes tensor transpose detection (handles both `[w=nc, h=anchors]` and `[w=anchors, h=nc]`)
- Fixed labels showing COCO names ("person", "bicycle") for custom finetune models
- Fixed picked labels being overridden by asset labels on model reload

### Improvements
- Labels auto-loaded from `labels.txt` in assets at startup
- Custom labels path persisted across model reloads
- Reset labels when switching back to bundled model
- C++ draw_detections now fully dynamic (uses custom_labels vector, falls back to COCO for 80-class, generic "C0" otherwise)

## 0.0.2

### New Features
- 🔀 **Multi-Model Support**: Auto-detects model format at load time
  - Format A: Models with baked-in post-processing (finetune/custom, 2 inputs)
  - Format C: Models without post-processing (COCO pretrained, 1 input, recommended)
- 🧠 **FPN Decode**: Full DFL softmax + expectation decode for no-postprocess models
- 🎛️ **Runtime Model Loading**: Pick .param and .bin separately from device storage
- ⚡ **GPU Toggle**: Switch CPU/GPU (Vulkan) from settings with auto-detection
- 🛡️ **Anti-Spoof**: FFT moiré pattern detection to reject screen/monitor images
- 📱 **Orientation Aware**: Camera preview rotates with device physical orientation
- 📷 **Capture Guard**: Photo capture only enabled when objects are detected
- ▶️ **Start/Stop Camera**: Manual control over detection pipeline

### Improvements
- Optimized pipeline: reduced from 4 bitmap copies to 1 per frame
- On-demand capture allocation (no per-frame overhead)
- Separate .param and .bin file picker buttons
- GPU availability check before enabling toggle
- `Resize` → `Interp` layer replacement documented

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
- Fixed num_class showing 1600 instead of 80 (min dimension heuristic)

### Model Support
- PP-PicoDet S/M/L (320×320)
- COCO 80-class pretrained models (no postprocess)
- Custom finetune models (any number of classes, with postprocess)
- ONNX → NCNN via PNNX conversion documented

## 0.0.1

- Initial project setup
- Basic NCNN inference pipeline
- JNI bridge + Kotlin plugin
- Dart API + method channel
