import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '虹膜检测 (Letterbox+NMS)',
      theme: ThemeData(useMaterial3: true),
      home: const MainDetectionPage(),
    );
  }
}

class MainDetectionPage extends StatefulWidget {
  const MainDetectionPage({super.key});

  @override
  State<MainDetectionPage> createState() => _MainDetectionPageState();
}

class _MainDetectionPageState extends State<MainDetectionPage> {
  Interpreter? _interpreter;
  List<CameraDescription>? _cameras;
  CameraController? _cameraController;
  int _selectedCameraIndex = 0;

  File? _staticImageFile;
  Uint8List? _processedImageBytes;
  final ImagePicker _picker = ImagePicker();

  bool _isProcessing = false;
  String _debugInfo = "初始化中...";
  List<Detection>? _liveDetections;
  Size? _liveImageSize;

  List<List<List<double>>>? _outputBuffer;

  @override
  void initState() {
    super.initState();
    _initApp();
  }

  Future<void> _initApp() async {
    final status = await [
      Permission.camera,
      Permission.storage,
      Permission.photos,
    ].request();

    if (status[Permission.camera] != PermissionStatus.granted) {
      setState(() {
        _debugInfo = "相机权限被拒绝，请在系统设置中开启后重试。";
      });
      return;
    }

    await _loadModel();
    _cameras = await availableCameras();
    if (_cameras != null && _cameras!.isNotEmpty) {
      _initCamera(_cameras![0]);
    } else {
      setState(() {
        _debugInfo = "未找到可用相机";
      });
    }
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float16.tflite');

      _outputBuffer ??=
          List.generate(1, (_) => List.generate(6, (_) => List.filled(8400, 0.0)));

      setState(() {
        _debugInfo = "模型加载成功，进入相机模式";
      });
    } catch (e) {
      setState(() => _debugInfo = "模型加载失败: $e");
    }
  }

  Future<void> _initCamera(CameraDescription cameraDescription) async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    _cameraController = CameraController(
      cameraDescription,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;

      _cameraController!.startImageStream((CameraImage image) {
        if (_staticImageFile == null && !_isProcessing) {
          _isProcessing = true;
          _runLiveInference(image);
        }
      });

      setState(() {});
    } catch (e) {
      setState(() {
        _debugInfo = "相机初始化失败: $e";
      });
    }
  }

  void _switchCamera() {
    if (_staticImageFile != null) return;
    if (_cameras == null || _cameras!.length < 2) return;
    _selectedCameraIndex = (_selectedCameraIndex + 1) % _cameras!.length;
    _initCamera(_cameras![_selectedCameraIndex]);
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() {
      _staticImageFile = File(image.path);
      _processedImageBytes = null;
      _debugInfo = "正在分析并绘制...";
    });

    compute(prepareStaticImage, image.path).then((data) {
      if (_interpreter == null) {
        throw Exception("模型尚未加载");
      }
      final Float32List inputTensor = data['tensor'] as Float32List;
      final Map<String, double> meta = Map<String, double>.from(data['meta']);
      var inputs = [inputTensor.reshape([1, 640, 640, 3])];
      var outputs = {0: _outputBuffer!};
      _interpreter!.runForMultipleInputs(inputs, outputs);

      return compute(drawOnStaticImage, [image.path, outputs[0], meta]);
    }).then((drawnImageBytes) {
      if (!mounted) return;
      setState(() {
        _processedImageBytes = drawnImageBytes;
        _debugInfo = "分析完成";
      });
    }).catchError((e) {
      if (!mounted) return;
      setState(() => _debugInfo = "失败: $e");
    });
  }

  void _clearImage() {
    setState(() {
      _staticImageFile = null;
      _processedImageBytes = null;
      _liveDetections = null;
      _liveImageSize = null;
      _debugInfo = "相机模式";
    });
  }

  Future<void> _runLiveInference(CameraImage image) async {
    final int startTime = DateTime.now().millisecondsSinceEpoch;
    try {
      if (_interpreter == null) return;

      final Map<String, dynamic> isolateData = {
        'yPlane': image.planes[0].bytes,
        'uPlane': image.planes[1].bytes,
        'vPlane': image.planes[2].bytes,
        'yRowStride': image.planes[0].bytesPerRow,
        'uvRowStride': image.planes[1].bytesPerRow,
        'uvPixelStride': image.planes[1].bytesPerPixel ?? 1,
        'width': image.width,
        'height': image.height,
      };

      final Map<String, dynamic> processed =
          await compute(cropAndConvert, isolateData);
      final Float32List inputTensor = processed['tensor'] as Float32List;
      final Map<String, double> meta = Map<String, double>.from(processed['meta']);

      var inputs = [inputTensor.reshape([1, 640, 640, 3])];
      var outputs = {0: _outputBuffer!};

      _interpreter!.runForMultipleInputs(inputs, outputs);

      var outputList = (outputs[0] as List)[0] as List<List<double>>;
      final detections = parseBoxes(outputList, 0.25, meta);

      final int cost = DateTime.now().millisecondsSinceEpoch - startTime;

      if (!mounted) return;

      setState(() {
        _liveDetections = detections.isNotEmpty ? detections : null;
        _liveImageSize = Size(meta['origWidth']!, meta['origHeight']!);
        _debugInfo = detections.isNotEmpty
            ? "🎯 锁定目标 (${cost} ms)"
            : "👀 搜索中... (${cost} ms)";
      });
    } catch (e) {
      setState(() {
        _debugInfo = "推理异常: $e";
      });
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('虹膜检测'),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            onPressed: _switchCamera,
          ),
        ],
      ),
      body: Stack(
        children: [
          if (_staticImageFile != null)
            Center(
              child: _processedImageBytes != null
                  ? Image.memory(_processedImageBytes!, fit: BoxFit.contain)
                  : const CircularProgressIndicator(),
            )
          else if (_cameraController != null &&
              _cameraController!.value.isInitialized)
            Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                if (_liveDetections != null && _liveImageSize != null)
                  CustomPaint(
                    painter: DetectionPainter(
                      _liveDetections!,
                      _liveImageSize!,
                    ),
                  ),
              ],
            )
          else
            const Center(child: CircularProgressIndicator()),
          Positioned(
            bottom: 100,
            left: 20,
            right: 20,
            child: Container(
              color: Colors.black54,
              padding: const EdgeInsets.all(8),
              child: Text(
                _debugInfo,
                style: const TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            ),
          )
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _staticImageFile == null ? _pickImage : _clearImage,
        child:
            Icon(_staticImageFile == null ? Icons.photo_library : Icons.close),
      ),
    );
  }
}

class Detection {
  Detection({
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
    required this.cls,
    required this.score,
  });

  final double cx;
  final double cy;
  final double w;
  final double h;
  final int cls;
  final double score;
}

Future<Map<String, dynamic>> prepareStaticImage(String path) async {
  final file = File(path);
  final Uint8List imageBytes = await file.readAsBytes();
  img.Image? originalImage = img.decodeImage(imageBytes);
  if (originalImage == null) throw Exception("解码失败");
  originalImage = img.bakeOrientation(originalImage);

  final letter = _letterbox(originalImage, 640, 640);
  final img.Image resized = letter['image'] as img.Image;
  final Map<String, double> meta = Map<String, double>.from(letter['meta']);

  var inputTensor = Float32List(1 * 640 * 640 * 3);
  int pixelIndex = 0;
  for (int y = 0; y < 640; y++) {
    for (int x = 0; x < 640; x++) {
      final pixel = resized.getPixel(x, y);
      inputTensor[pixelIndex++] = pixel.r / 255.0;
      inputTensor[pixelIndex++] = pixel.g / 255.0;
      inputTensor[pixelIndex++] = pixel.b / 255.0;
    }
  }
  return {'tensor': inputTensor, 'meta': meta};
}

Future<Uint8List> drawOnStaticImage(List<dynamic> args) async {
  String path = args[0] as String;
  List outputData = args[1] as List;
  Map<String, double> meta = Map<String, double>.from(args[2] as Map);
  var outputList = outputData[0] as List<List<double>>;

  final Uint8List imageBytes = await File(path).readAsBytes();
  img.Image? image = img.decodeImage(imageBytes);
  if (image == null) throw Exception("无法读取原图");
  image = img.bakeOrientation(image);

  img.BitmapFont font = img.arial24;

  final detections = parseBoxes(outputList, 0.25, meta);

  for (final det in detections) {
    double normD = (det.w + det.h) / 2.0;
    int radius = (normD / 2).round();
    int x = det.cx.round();
    int y = det.cy.round();

    img.Color color;
    String label;
    String scoreStr = det.score.toStringAsFixed(3);

    if (det.cls == 0) {
      color = img.ColorRgb8(255, 0, 0);
      label = "irisOuter $scoreStr";
    } else {
      color = img.ColorRgb8(0, 255, 255);
      label = "irisInner $scoreStr";
    }

    img.drawCircle(image, x: x, y: y, radius: radius, color: color);
    img.drawCircle(image, x: x, y: y, radius: radius - 1, color: color);
    img.drawCircle(image, x: x, y: y, radius: radius + 1, color: color);

    int textWidth = label.length * 14;
    int textHeight = 24;
    int bgX = x - textWidth ~/ 2;
    int bgY = y - radius - textHeight - 5;

    img.fillRect(
      image,
      x1: bgX,
      y1: bgY,
      x2: bgX + textWidth,
      y2: bgY + textHeight,
      color: img.ColorRgba8(0, 0, 0, 150),
    );
    img.drawString(
      image,
      label,
      font: font,
      x: bgX + 2,
      y: bgY + 2,
      color: img.ColorRgb8(255, 255, 255),
    );
  }

  return img.encodeJpg(image);
}

Map<String, dynamic> cropAndConvert(Map<String, dynamic> data) {
  final Uint8List yPlane = data['yPlane'];
  final Uint8List uPlane = data['uPlane'];
  final Uint8List vPlane = data['vPlane'];
  final int yRowStride = data['yRowStride'];
  final int uvRowStride = data['uvRowStride'];
  final int uvPixelStride = data['uvPixelStride'];
  final int width = data['width'];
  final int height = data['height'];

  var imgBuffer = img.Image(width: width, height: height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      final int uvIndex =
          uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final int index = y * yRowStride + x;

      if (index >= yPlane.length || uvIndex >= uPlane.length) continue;

      final yp = yPlane[index];
      final up = uPlane[uvIndex];
      final vp = vPlane[uvIndex];

      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp -
              up * 46549 / 131072 +
              44 -
              vp * 93604 / 131072 +
              91)
          .round()
          .clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

      imgBuffer.setPixelRgb(x, y, r, g, b);
    }
  }

  if (imgBuffer.width > imgBuffer.height) {
    imgBuffer = img.copyRotate(imgBuffer, angle: 90);
  }

  final letter = _letterbox(imgBuffer, 640, 640);
  final img.Image resized = letter['image'] as img.Image;
  final Map<String, double> meta = Map<String, double>.from(letter['meta']);

  var output = Float32List(1 * 640 * 640 * 3);
  int pixelIndex = 0;
  for (int y = 0; y < 640; y++) {
    for (int x = 0; x < 640; x++) {
      final pixel = resized.getPixel(x, y);
      output[pixelIndex++] = pixel.r / 255.0;
      output[pixelIndex++] = pixel.g / 255.0;
      output[pixelIndex++] = pixel.b / 255.0;
    }
  }
  return {'tensor': output, 'meta': meta};
}

Map<String, dynamic> _letterbox(img.Image src, int targetW, int targetH) {
  final double scale = math.min(targetW / src.width, targetH / src.height);
  final int newW = (src.width * scale).round();
  final int newH = (src.height * scale).round();

  final img.Image resized = img.copyResize(src, width: newW, height: newH);
  final img.Image padded = img.Image(width: targetW, height: targetH);

  padded.fill(0x000000); // black padding
  final int padX = ((targetW - newW) / 2).floor();
  final int padY = ((targetH - newH) / 2).floor();
  img.copyInto(padded, resized, dstX: padX, dstY: padY);

  return {
    'image': padded,
    'meta': {
      'scale': scale,
      'padX': padX.toDouble(),
      'padY': padY.toDouble(),
      'origWidth': src.width.toDouble(),
      'origHeight': src.height.toDouble(),
    }
  };
}

List<Detection> parseBoxes(
    List<List<double>> outputList, double threshold, Map<String, double> meta) {
  final double scale = meta['scale'] ?? 1.0;
  final double padX = meta['padX'] ?? 0.0;
  final double padY = meta['padY'] ?? 0.0;

  List<Detection> candidates = [];

  for (int i = 0; i < 8400; i++) {
    double scoreOuter = outputList[4][i];
    double scoreInner = outputList[5][i];

    if (scoreOuter > threshold) {
      final det = _buildDetection(outputList, i, 0, scoreOuter, scale, padX, padY);
      candidates.add(det);
    }

    if (scoreInner > threshold) {
      final det = _buildDetection(outputList, i, 1, scoreInner, scale, padX, padY);
      candidates.add(det);
    }
  }

  return _nonMaxSuppression(candidates, 0.45);
}

Detection _buildDetection(List<List<double>> output, int idx, int cls, double score,
    double scale, double padX, double padY) {
  double cx640 = output[0][idx] * 640;
  double cy640 = output[1][idx] * 640;
  double w640 = output[2][idx] * 640;
  double h640 = output[3][idx] * 640;

  double cx = (cx640 - padX) / scale;
  double cy = (cy640 - padY) / scale;
  double w = w640 / scale;
  double h = h640 / scale;

  return Detection(cx: cx, cy: cy, w: w, h: h, cls: cls, score: score);
}

List<Detection> _nonMaxSuppression(List<Detection> detections, double iouThresh) {
  detections.sort((a, b) => b.score.compareTo(a.score));
  List<Detection> result = [];
  List<bool> removed = List.filled(detections.length, false);

  for (int i = 0; i < detections.length; i++) {
    if (removed[i]) continue;
    final current = detections[i];
    result.add(current);
    for (int j = i + 1; j < detections.length; j++) {
      if (removed[j]) continue;
      if (detections[j].cls != current.cls) continue;
      final iou = _bboxIou(current, detections[j]);
      if (iou > iouThresh) {
        removed[j] = true;
      }
    }
  }
  return result;
}

double _bboxIou(Detection a, Detection b) {
  final double ax1 = a.cx - a.w / 2;
  final double ay1 = a.cy - a.h / 2;
  final double ax2 = a.cx + a.w / 2;
  final double ay2 = a.cy + a.h / 2;

  final double bx1 = b.cx - b.w / 2;
  final double by1 = b.cy - b.h / 2;
  final double bx2 = b.cx + b.w / 2;
  final double by2 = b.cy + b.h / 2;

  final double interX1 = math.max(ax1, bx1);
  final double interY1 = math.max(ay1, by1);
  final double interX2 = math.min(ax2, bx2);
  final double interY2 = math.min(ay2, by2);

  final double interW = math.max(0, interX2 - interX1);
  final double interH = math.max(0, interY2 - interY1);
  final double interArea = interW * interH;

  final double unionArea = a.w * a.h + b.w * b.h - interArea;
  if (unionArea <= 0) return 0.0;
  return interArea / unionArea;
}

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size imageSize;

  DetectionPainter(this.detections, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    final double scale = size.width / imageSize.width;
    final double drawHeight = imageSize.height * scale;
    final double offsetY = (size.height - drawHeight) / 2;

    for (final det in detections) {
      double radius = ((det.w + det.h) / 4.0) * scale;
      double x = det.cx * scale;
      double y = det.cy * scale + offsetY;

      Paint paint;
      String label;
      String scoreStr = det.score.toStringAsFixed(3);

      if (det.cls == 0) {
        paint = Paint()
          ..color = Colors.redAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0;
        label = "外圆 $scoreStr";
      } else {
        paint = Paint()
          ..color = Colors.cyanAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0;
        label = "内圆 $scoreStr";
      }

      canvas.drawCircle(Offset(x, y), radius, paint);

      TextSpan span = TextSpan(
        style: TextStyle(
          color: paint.color,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black54,
        ),
        text: label,
      );
      TextPainter tp = TextPainter(
        text: span,
        textAlign: TextAlign.left,
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, Offset(x - tp.width / 2, y - radius - 25));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
