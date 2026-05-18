import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:fast_paddle_detection/paddle_detection.dart';
import 'package:permission_handler/permission_handler.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'PicoDet Demo',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
    darkTheme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true, brightness: Brightness.dark),
    home: const HomePage(),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _detector = PaddleDetection();
  bool _modelLoaded = false;
  bool _loading = false;
  String _status = 'Loading model...';
  double _threshold = 0.8;
  int _numClass = 0;
  String? _customParamPath, _customBinPath, _customLabelsPath;
  bool get _useCustom => _customParamPath != null && _customBinPath != null;
  int _tabIndex = 0;
  bool _gpuAvailable = false;
  bool _useGpu = false;
  bool _antiSpoof = false; // default CPU

  @override
  void initState() {
    super.initState();
    _checkGpuAndLoad();
  }

  Future<void> _checkGpuAndLoad() async {
    _gpuAvailable = await _detector.hasGpu();
    _loadModel();
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() {
      _loading = true;
      _status = 'Loading...';
    });
    try {
      final cpuGpu = _useGpu ? 1 : 0;
      final info = _useCustom
          ? await _detector.loadModelFromFile(paramPath: _customParamPath!, binPath: _customBinPath!, cpuGpu: cpuGpu)
          : await _detector.loadModel(
              paramName: 'model_no_post_pro.ncnn.param',
              binName: 'model_no_post_pro.ncnn.bin',
              cpuGpu: cpuGpu,
            );
      if (info.success) {
        await _detector.setThreshold(threshold: _threshold);
        // Load labels: custom file if picked, otherwise try asset
        if (_customLabelsPath != null) {
          try {
            await _detector.loadLabelsFromFile(_customLabelsPath!);
          } catch (_) {}
        } else {
          try {
            await _detector.loadLabelsFromAsset('labels.txt');
          } catch (_) {}
        }
      }
      final backend = _useGpu ? 'GPU' : 'CPU';
      setState(() {
        _modelLoaded = info.success;
        _numClass = info.numClass;
        _loading = false;
        _status = info.success ? 'Model loaded ✓ (${info.numClass} cls, $backend)' : 'Load failed';
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _status = 'Error: $e';
      });
    }
  }

  Future<void> _applyThreshold() async {
    if (_modelLoaded) await _detector.setThreshold(threshold: _threshold);
  }

  Future<void> _pickParam() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any, allowMultiple: false);
    if (result == null || result.files.isEmpty || result.files.first.path == null) return;
    setState(() {
      _customParamPath = result.files.first.path;
      _modelLoaded = false;
    });
    _snack('Param: ${result.files.first.name}');
  }

  Future<void> _pickBin() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any, allowMultiple: false);
    if (result == null || result.files.isEmpty || result.files.first.path == null) return;
    setState(() {
      _customBinPath = result.files.first.path;
      _modelLoaded = false;
    });
    _snack('Bin: ${result.files.first.name}');
  }

  Future<void> _reloadCustomModel() async {
    if (_customParamPath == null || _customBinPath == null) {
      _snack('Pick both .param and .bin first');
      return;
    }
    _loadModel();
  }

  Future<void> _pickLabels() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any, allowMultiple: false);
    if (result == null || result.files.isEmpty || result.files.first.path == null) return;
    _customLabelsPath = result.files.first.path;
    try {
      final labels = await _detector.loadLabelsFromFile(_customLabelsPath!);
      _snack('Labels loaded: ${labels.length} (${labels.take(3).join(", ")}...)');
    } catch (e) {
      _snack('Failed: $e');
      _customLabelsPath = null;
    }
  }

  void _useBundled() {
    setState(() {
      _customParamPath = null;
      _customBinPath = null;
      _customLabelsPath = null;
      _modelLoaded = false;
    });
    _loadModel();
  }

  void _snack(String m) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m)));

  void _openSettings() => showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    builder: (_) => _SettingsSheet(
      threshold: _threshold,
      useCustom: _useCustom,
      customParam: _customParamPath,
      customBin: _customBinPath,
      modelLoaded: _modelLoaded,
      loading: _loading,
      numClass: _numClass,
      gpuAvailable: _gpuAvailable,
      useGpu: _useGpu,
      antiSpoof: _antiSpoof,
      onThresholdChanged: (v) {
        setState(() => _threshold = v);
        _applyThreshold();
      },
      onGpuChanged: (v) {
        setState(() => _useGpu = v);
        _loadModel(); // reload with new backend
      },
      onAntiSpoofChanged: (v) {
        setState(() => _antiSpoof = v);
        _detector.setAntiSpoof(enabled: v);
      },
      onPickModel: _pickParam,
      onPickBin: _pickBin,
      onPickLabels: _pickLabels,
      onUseBundled: _useBundled,
      onReload: () {
        Navigator.pop(context);
        _reloadCustomModel();
      },
    ),
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('PicoDet'),
      centerTitle: true,
      actions: [IconButton(icon: const Icon(Icons.settings), onPressed: _openSettings)],
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(22),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              _modelLoaded ? Icons.check_circle : Icons.info_outline,
              size: 14,
              color: _modelLoaded ? Colors.green : Colors.orange,
            ),
            const SizedBox(width: 6),
            Text(_status, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    ),
    body: IndexedStack(
      index: _tabIndex,
      children: [
        _CameraTab(detector: _detector, modelLoaded: _modelLoaded),
        _GalleryTab(detector: _detector, modelLoaded: _modelLoaded),
      ],
    ),
    bottomNavigationBar: NavigationBar(
      selectedIndex: _tabIndex,
      onDestinationSelected: (i) => setState(() => _tabIndex = i),
      destinations: const [
        NavigationDestination(icon: Icon(Icons.videocam), label: 'Camera'),
        NavigationDestination(icon: Icon(Icons.photo_library), label: 'Gallery'),
      ],
    ),
  );
}

// =============================================================================
// Camera Tab
// =============================================================================
class _CameraTab extends StatefulWidget {
  final PaddleDetection detector;
  final bool modelLoaded;
  const _CameraTab({required this.detector, required this.modelLoaded});
  @override
  State<_CameraTab> createState() => _CameraTabState();
}

class _CameraTabState extends State<_CameraTab> {
  int? _textureId;
  int _previewW = 0, _previewH = 0;
  List<DetectionResult> _detections = [];
  StreamSubscription? _sub;
  bool _started = false;
  bool _flashOn = false;
  bool _capturing = false;
  int _fps = 0;
  int _frameCount = 0;
  int _inferenceMs = 0;
  int _deviceRotation = 0;
  Timer? _fpsTimer;

  /// Convert device rotation degrees to RotatedBox quarter turns.
  /// deviceRotation: 0=portrait, 90=landscape-left, 180=upside-down, 270=landscape-right
  int get _quarterTurns {
    switch (_deviceRotation) {
      case 90:
        return 1; // landscape left
      case 180:
        return 2; // upside down
      case 270:
        return 3; // landscape right
      default:
        return 0; // portrait
    }
  }

  @override
  void didUpdateWidget(covariant _CameraTab old) {
    super.didUpdateWidget(old);
  }

  @override
  void initState() {
    super.initState();
  }

  Future<void> _start() async {
    if (_started) return;
    final status = await Permission.camera.request();
    if (!status.isGranted) return;
    try {
      final info = await widget.detector.startCamera();
      _sub = widget.detector.detectionStream.listen((event) {
        if (mounted) {
          _frameCount++;
          setState(() {
            _detections = event.detections;
            _inferenceMs = event.inferenceMs;
            _deviceRotation = event.deviceRotation;
            if (event.imageWidth > 0 && event.imageHeight > 0) {
              _previewW = event.imageWidth;
              _previewH = event.imageHeight;
            }
          });
        }
      });
      _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (mounted)
          setState(() {
            _fps = _frameCount;
            _frameCount = 0;
          });
      });
      if (mounted)
        setState(() {
          _textureId = info.textureId;
          _previewW = info.previewWidth;
          _previewH = info.previewHeight;
          _started = true;
        });
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  Future<void> _stop() async {
    _sub?.cancel();
    _sub = null;
    _fpsTimer?.cancel();
    _fpsTimer = null;
    await widget.detector.stopCamera();
    if (mounted)
      setState(() {
        _textureId = null;
        _started = false;
        _detections = [];
        _flashOn = false;
        _fps = 0;
        _frameCount = 0;
      });
  }

  Future<void> _toggleFlash() async {
    final on = await widget.detector.toggleFlash(enable: !_flashOn);
    setState(() => _flashOn = on);
  }

  Future<void> _capture() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    try {
      // Save to app's external files directory
      final dir = Directory('/storage/emulated/0/DCIM/PicoDet');
      final result = await widget.detector.capturePhoto(folder: dir.path, prefix: 'det');
      if (mounted) {
        _showCaptureResult(result);
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Capture error: $e')));
    }
    setState(() => _capturing = false);
  }

  void _showCaptureResult(CaptureResult result) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.75,
        minChildSize: 0.4,
        maxChildSize: 0.95,
        builder: (ctx, scroll) => ListView(
          controller: scroll,
          padding: const EdgeInsets.all(16),
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(color: Colors.grey[400], borderRadius: BorderRadius.circular(2)),
              ),
            ),
            const Text('Capture Result', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            // Two images side by side
            Row(
              children: [
                Expanded(
                  child: Column(
                    children: [
                      const Text('With BBox', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      GestureDetector(
                        onTap: () => _showFullImage(ctx, result.annotatedPath),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(File(result.annotatedPath), fit: BoxFit.contain, height: 180),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    children: [
                      const Text('Clean', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      GestureDetector(
                        onTap: () => _showFullImage(ctx, result.cleanPath),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.file(File(result.cleanPath), fit: BoxFit.contain, height: 180),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            const Text(
              'Tap image for full screen',
              style: TextStyle(fontSize: 11, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text('Detections: ${result.detections.length}', style: const TextStyle(fontWeight: FontWeight.bold)),
            ...result.detections.map(
              (d) => ListTile(
                dense: true,
                leading: CircleAvatar(
                  radius: 12,
                  backgroundColor: _Cls.color(d.label),
                  child: Text(
                    _Cls.name(d.label),
                    style: const TextStyle(fontSize: 8, color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                ),
                title: Text('${_Cls.name(d.label)} — ${(d.probability * 100).toStringAsFixed(1)}%'),
                subtitle: Text(
                  '${d.x.toStringAsFixed(0)},${d.y.toStringAsFixed(0)} ${d.width.toStringAsFixed(0)}×${d.height.toStringAsFixed(0)}',
                ),
              ),
            ),
            const Divider(),
            Text('Clean: ${result.cleanPath}', style: const TextStyle(fontSize: 10, fontFamily: 'monospace')),
            Text('Bbox:  ${result.annotatedPath}', style: const TextStyle(fontSize: 10, fontFamily: 'monospace')),
          ],
        ),
      ),
    );
  }

  void _showFullImage(BuildContext ctx, String path) {
    Navigator.of(ctx).push(
      MaterialPageRoute(
        builder: (_) => Scaffold(
          backgroundColor: Colors.black,
          appBar: AppBar(backgroundColor: Colors.black, foregroundColor: Colors.white, elevation: 0),
          body: Center(
            child: InteractiveViewer(child: Image.file(File(path), fit: BoxFit.contain)),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _fpsTimer?.cancel();
    widget.detector.stopCamera();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.modelLoaded)
      return const Center(
        child: Text('Loading model...', style: TextStyle(color: Colors.grey, fontSize: 16)),
      );

    // Camera not started — show start button
    if (!_started || _textureId == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.videocam_off, size: 64, color: Colors.grey[400]),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _start,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Start Detection'),
            ),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Rotate preview to match physical device orientation
        Center(
          child: RotatedBox(
            quarterTurns: _quarterTurns,
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _previewW.toDouble(),
                height: _previewH.toDouble(),
                child: Texture(textureId: _textureId!),
              ),
            ),
          ),
        ),
        // Top info (FPS only in debug mode — this is Flutter overlay, not drawn on image)
        if (kDebugMode)
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(8)),
              child: Text(
                '$_fps FPS • ${_inferenceMs}ms • ${_detections.length} obj',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        // Bottom controls
        Positioned(
          bottom: 16,
          left: 0,
          right: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // Flash
              FloatingActionButton.small(
                heroTag: 'flash',
                onPressed: _toggleFlash,
                backgroundColor: _flashOn ? Colors.amber : Colors.black54,
                child: Icon(_flashOn ? Icons.flash_on : Icons.flash_off, color: Colors.white),
              ),
              // Capture — disabled when no objects detected
              FloatingActionButton(
                heroTag: 'capture',
                onPressed: (_capturing || _detections.isEmpty) ? null : _capture,
                backgroundColor: _detections.isEmpty ? Colors.grey : Colors.white,
                child: _capturing
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                    : Icon(Icons.camera, color: _detections.isEmpty ? Colors.white54 : Colors.black, size: 32),
              ),
              // Stop camera
              FloatingActionButton.small(
                heroTag: 'stop',
                onPressed: _stop,
                backgroundColor: Colors.red.shade700,
                child: const Icon(Icons.stop, color: Colors.white),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// =============================================================================
// Gallery Tab
// =============================================================================
class _GalleryTab extends StatefulWidget {
  final PaddleDetection detector;
  final bool modelLoaded;
  const _GalleryTab({required this.detector, required this.modelLoaded});
  @override
  State<_GalleryTab> createState() => _GalleryTabState();
}

class _GalleryTabState extends State<_GalleryTab> {
  final _picker = ImagePicker();
  String? _imagePath;
  Size? _imageSize;
  List<DetectionResult> _detections = [];
  int _ms = 0;
  bool _busy = false;

  Future<void> _pick() async {
    if (!widget.modelLoaded) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Model not loaded')));
      return;
    }
    final f = await _picker.pickImage(source: ImageSource.gallery);
    if (f == null) return;
    setState(() {
      _imagePath = f.path;
      _detections = [];
      _busy = true;
    });
    final sz = await _imgSize(f.path);
    final sw = Stopwatch()..start();
    final r = await widget.detector.detect(f.path);
    sw.stop();
    setState(() {
      _imageSize = sz;
      _detections = r;
      _ms = sw.elapsedMilliseconds;
      _busy = false;
    });
  }

  Future<Size> _imgSize(String p) async {
    final d = await File(p).readAsBytes();
    final c = await ui.instantiateImageCodec(d);
    final fr = await c.getNextFrame();
    final s = Size(fr.image.width.toDouble(), fr.image.height.toDouble());
    fr.image.dispose();
    return s;
  }

  @override
  Widget build(BuildContext context) => Column(
    children: [
      Expanded(
        child: _imagePath == null
            ? Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.photo_library_outlined, size: 64, color: Colors.grey[400]),
                    const SizedBox(height: 12),
                    const Text('Pick an image', style: TextStyle(color: Colors.grey)),
                  ],
                ),
              )
            : Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(File(_imagePath!), fit: BoxFit.contain),
                  if (_detections.isNotEmpty && _imageSize != null)
                    LayoutBuilder(
                      builder: (_, c) => CustomPaint(
                        painter: _StaticPainter(
                          imageSize: _imageSize!,
                          detections: _detections,
                          displaySize: Size(c.maxWidth, c.maxHeight),
                        ),
                      ),
                    ),
                  if (_busy)
                    Container(
                      color: Colors.black26,
                      child: const Center(child: CircularProgressIndicator()),
                    ),
                ],
              ),
      ),
      if (_detections.isNotEmpty)
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          child: Row(
            children: [
              Text('${_detections.length} obj', style: const TextStyle(fontWeight: FontWeight.bold)),
              const Spacer(),
              Text('${_ms}ms', style: const TextStyle(fontFamily: 'monospace')),
            ],
          ),
        ),
      if (_detections.isNotEmpty)
        SizedBox(
          height: 90,
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            itemCount: _detections.length,
            itemBuilder: (_, i) {
              final d = _detections[i];
              return ListTile(
                dense: true,
                leading: CircleAvatar(
                  radius: 14,
                  backgroundColor: _Cls.color(d.label),
                  child: Text(
                    _Cls.name(d.label),
                    style: const TextStyle(fontSize: 9, color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                ),
                title: Text('${_Cls.name(d.label)} ${(d.probability * 100).toStringAsFixed(1)}%'),
                subtitle: Text(
                  '${d.x.toStringAsFixed(0)},${d.y.toStringAsFixed(0)} ${d.width.toStringAsFixed(0)}×${d.height.toStringAsFixed(0)}',
                ),
              );
            },
          ),
        ),
      Padding(
        padding: const EdgeInsets.all(12),
        child: SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            onPressed: _busy ? null : _pick,
            icon: const Icon(Icons.image_search),
            label: const Text('Pick & Detect'),
          ),
        ),
      ),
    ],
  );
}

// =============================================================================
// Settings
// =============================================================================
class _SettingsSheet extends StatefulWidget {
  final double threshold;
  final bool useCustom;
  final String? customParam, customBin;
  final bool modelLoaded, loading;
  final int numClass;
  final bool gpuAvailable;
  final bool useGpu;
  final bool antiSpoof;
  final ValueChanged<double> onThresholdChanged;
  final ValueChanged<bool> onGpuChanged;
  final ValueChanged<bool> onAntiSpoofChanged;
  final VoidCallback onPickModel, onPickBin, onPickLabels, onUseBundled, onReload;
  const _SettingsSheet({
    required this.threshold,
    required this.useCustom,
    required this.customParam,
    required this.customBin,
    required this.modelLoaded,
    required this.loading,
    required this.numClass,
    required this.gpuAvailable,
    required this.useGpu,
    required this.antiSpoof,
    required this.onThresholdChanged,
    required this.onGpuChanged,
    required this.onAntiSpoofChanged,
    required this.onPickModel,
    required this.onPickBin,
    required this.onPickLabels,
    required this.onUseBundled,
    required this.onReload,
  });
  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  late double _tr;
  late bool _gpu;
  late bool _spoof;
  @override
  void initState() {
    super.initState();
    _tr = widget.threshold;
    _gpu = widget.useGpu;
    _spoof = widget.antiSpoof;
  }

  @override
  Widget build(BuildContext context) => DraggableScrollableSheet(
    expand: false,
    initialChildSize: 0.45,
    minChildSize: 0.3,
    maxChildSize: 0.7,
    builder: (_, scroll) => ListView(
      controller: scroll,
      padding: const EdgeInsets.all(20),
      children: [
        Center(
          child: Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(color: Colors.grey[400], borderRadius: BorderRadius.circular(2)),
          ),
        ),
        const SizedBox(height: 16),
        Text('Settings', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 16),
        Text('Threshold: ${_tr.toStringAsFixed(2)}', style: const TextStyle(fontWeight: FontWeight.bold)),
        const Text('Only detections ≥ this score are shown', style: TextStyle(fontSize: 12, color: Colors.grey)),
        Slider(
          value: _tr,
          min: 0.1,
          max: 1.0,
          divisions: 90,
          onChanged: (v) {
            setState(() => _tr = v);
            widget.onThresholdChanged(v);
          },
        ),
        const Divider(),
        // GPU toggle
        Row(
          children: [
            const Text('Compute: ', style: TextStyle(fontWeight: FontWeight.bold)),
            const Spacer(),
            Text(
              widget.gpuAvailable ? (_gpu ? 'GPU (Vulkan)' : 'CPU') : 'CPU only (no GPU)',
              style: const TextStyle(fontSize: 12),
            ),
            const SizedBox(width: 8),
            Switch(
              value: _gpu,
              onChanged: widget.gpuAvailable
                  ? (v) {
                      setState(() => _gpu = v);
                      widget.onGpuChanged(v);
                    }
                  : null,
            ),
          ],
        ),
        if (!widget.gpuAvailable)
          const Text('GPU not available on this device', style: TextStyle(fontSize: 11, color: Colors.grey)),
        const Divider(),
        // Anti-spoof toggle
        Row(
          children: [
            const Text('Anti-Spoof: ', style: TextStyle(fontWeight: FontWeight.bold)),
            const Spacer(),
            Text(_spoof ? 'ON' : 'OFF', style: const TextStyle(fontSize: 12)),
            const SizedBox(width: 8),
            Switch(
              value: _spoof,
              onChanged: (v) {
                setState(() => _spoof = v);
                widget.onAntiSpoofChanged(v);
              },
            ),
          ],
        ),
        const Text(
          'Skip detection if image is from screen/monitor',
          style: TextStyle(fontSize: 11, color: Colors.grey),
        ),
        const Divider(),
        Text('Classes from model: ${widget.numClass}', style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Text(
          widget.useCustom
              ? 'param: ${widget.customParam?.split('/').last ?? "not set"}\nbin: ${widget.customBin?.split('/').last ?? "not set"}'
              : 'Bundled model',
          style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: widget.loading ? null : widget.onPickModel,
                icon: const Icon(Icons.description, size: 16),
                label: const Text('.param'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: widget.loading ? null : widget.onPickBin,
                icon: const Icon(Icons.data_object, size: 16),
                label: const Text('.bin'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: widget.loading ? null : widget.onPickLabels,
          icon: const Icon(Icons.label_outline, size: 16),
          label: const Text('Pick labels.txt'),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: widget.loading ? null : widget.onReload,
                icon: const Icon(Icons.refresh),
                label: const Text('Load Custom'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton(
                onPressed: (widget.loading || !widget.useCustom) ? null : widget.onUseBundled,
                child: const Text('Use Bundled'),
              ),
            ),
          ],
        ),
      ],
    ),
  );
}

// =============================================================================
// Painters
// =============================================================================
class _StaticPainter extends CustomPainter {
  final Size imageSize;
  final List<DetectionResult> detections;
  final Size displaySize;
  _StaticPainter({required this.imageSize, required this.detections, required this.displaySize});
  @override
  void paint(Canvas canvas, Size size) {
    final sx = displaySize.width / imageSize.width;
    final sy = displaySize.height / imageSize.height;
    final s = sx < sy ? sx : sy;
    final ox = (displaySize.width - imageSize.width * s) / 2;
    final oy = (displaySize.height - imageSize.height * s) / 2;
    for (final d in detections) {
      final c = _Cls.color(d.label);
      final rect = Rect.fromLTWH(ox + d.x * s, oy + d.y * s, d.width * s, d.height * s);
      canvas.drawRect(
        rect,
        Paint()
          ..color = c
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );
      final label = '${_Cls.name(d.label)} ${(d.probability * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(color: Colors.white, fontSize: 12),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final bg = Rect.fromLTWH(rect.left, rect.top - tp.height - 4, tp.width + 8, tp.height + 4);
      canvas.drawRect(bg, Paint()..color = c);
      tp.paint(canvas, Offset(bg.left + 4, bg.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _StaticPainter old) => old.detections != detections || old.imageSize != imageSize;
}

// =============================================================================
class _Cls {
  static const _n = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'LCK',
    'SCR',
  ];
  static const _c = [
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal, Colors.cyan, Colors.blue, Colors.indigo, Colors.purple,
    Colors.red, Colors.pink, Colors.orange, Colors.amber, Colors.yellow, Colors.lime,
    Colors.green, Colors.teal,
    Colors.green, Colors.purple, // LCK, SCR
  ];
  static String name(int l) => l >= 0 && l < _n.length ? _n[l] : 'C$l';
  static Color color(int l) => l >= 0 && l < _c.length ? _c[l] : Colors.grey;
}
