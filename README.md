# TinyML_project
TinyML Sign Language Recognition (Edge AI)

This project implements a TinyML-based hand sign recognition system optimized for real-time inference on edge devices, with a focus on small model size, low latency, and deployability on Google Coral / EdgeTPU.

The project covers the full pipeline from data collection to deployment:

Hand landmark extraction using MediaPipe

Feature vector construction

Lightweight MLP model training

INT8 TensorFlow Lite conversion

Real-time inference on laptop and Coral devices

Project Structure

AI HNL/
├── capture_dataset.py – Capture hand landmarks and build dataset
├── train_mlp.py – Train lightweight MLP classifier
├── convert_tflite.py – Convert trained model to INT8 TFLite
├── run_realtime.py – Real-time inference (Laptop / Coral)
│
├── models/
│ └── mlp48_int8.tflite – Quantized INT8 model (~13 KB)
│
├── data_vec/ – Generated feature vectors (ignored in Git)
├── README.md
└── .gitignore

Technologies Used

Python 3.x

MediaPipe Hands

TensorFlow / Keras

TensorFlow Lite (INT8 quantization)

Google Coral / EdgeTPU

OpenCV

System Workflow

Step 1: Dataset Collection
Hand landmarks are extracted from a live camera feed using MediaPipe Hands and converted into numerical feature vectors.

Command:
python capture_dataset.py

Output:

Feature vectors saved in data_vec/

Corresponding class labels

Step 2: Model Training
A lightweight Multi-Layer Perceptron (MLP) is trained using the generated feature vectors.

Command:
python train_mlp.py

Output:

Trained Keras model (.keras format)

Step 3: INT8 Model Conversion
The trained model is quantized to INT8 TensorFlow Lite to reduce size and improve inference speed on edge devices.

Command:
python convert_tflite.py

Output:

mlp48_int8.tflite (~13 KB)

Step 4: Real-Time Inference
Run real-time hand sign recognition from camera input.

Command:
python run_realtime.py

Supported platforms:

Laptop (CPU inference)

Google Coral Dev Board (EdgeTPU inference)

Performance Summary

Platform: Laptop (CPU)
Approximate FPS: ~30
Notes: Stable real-time performance

Platform: Coral Dev Board
Approximate FPS: ~12–15
Notes: EdgeTPU inference with camera overhead

Additional details:

Model size: approximately 13 KB

Quantization: INT8

Suitable for embedded and edge deployment

Limitations

MediaPipe landmark accuracy depends heavily on lighting and hand visibility

Similar hand poses may lead to low-confidence predictions

Single-hand recognition only

Camera position and hand distance require calibration

No temporal smoothing between frames

Future Improvements

Confidence threshold auto-calibration

Temporal smoothing across consecutive frames

Multi-hand and sentence-level recognition

Camera-independent landmark normalization

Improved EdgeTPU delegate optimization

Git and Repository Notes

data_vec/ is excluded from Git to avoid pushing large generated data

The model file is tracked due to its small size and reproducibility

Repository is designed for academic and experimental use

Author

Phan Nam
TinyML – Edge AI – Computer Vision
Academic and experimental project

License

This project is intended for educational and research purposes only.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


User Guidance

This section provides practical guidance for users to correctly operate the system and obtain reliable hand sign recognition results.

Camera Setup and Environment

Use a stable camera (laptop webcam or USB camera) positioned at approximately chest height.

Ensure sufficient and uniform lighting; avoid strong backlight or shadows.

The background should be simple and uncluttered to reduce landmark detection noise.

Maintain a fixed distance between the hand and the camera (recommended: 40–70 cm).

Hand Positioning and Orientation

Only one hand should be visible in the camera frame at a time.

Keep the hand fully within the camera view; avoid partial occlusion.

The palm should generally face the camera unless the trained sign explicitly requires a different orientation.

Avoid extreme rotations or tilting of the hand, especially during real-time inference.

Gesture Execution

Perform gestures slowly and clearly, holding each sign steady for a short moment.

Avoid rapid transitions between gestures to prevent unstable predictions.

Keep fingers separated and clearly articulated according to the trained sign definitions.

If misclassification occurs, slightly adjust hand distance or orientation and retry.

Real-Time Inference Usage

Start the real-time recognition script only after the camera feed is stable.

Observe the predicted label and confidence score (if displayed).

If predictions fluctuate, pause movement briefly to allow the model to stabilize.

For best results, ensure the same camera viewpoint used during dataset collection.

Coral Deployment Notes

When running on Google Coral Dev Board, ensure the EdgeTPU delegate is correctly loaded.

Expect lower FPS compared to laptop CPU due to camera I/O and display overhead.

If no display is connected, external monitoring or predefined hand positioning may be required.

Threshold calibration may be necessary to reduce false positives in edge deployment.

Common Issues and Troubleshooting

Low confidence predictions: Improve lighting, reduce background clutter, and stabilize hand pose.

Incorrect classification: Re-align hand orientation or move closer/farther from the camera.

No detection: Ensure the hand is fully visible and MediaPipe is detecting landmarks correctly.

Performance drops: Reduce camera resolution or close background applications.

Recommended Usage Scope

Educational demonstrations of TinyML and Edge AI

Research experiments in hand gesture recognition

Prototyping for assistive or embedded systems

The system is not intended for safety-critical or commercial-grade deployment without further validation and robustness improvements.

-------------------------------------------------------------------------------------------------------------------------------------------

Running on Coral (Minimal Deployment)

This Coral deployment contains only:

stream_cam_predict.py

models/mlp48_int8.tflite (INT8 quantized TFLite model)

Prerequisites

Python 3 installed on the Coral device

A working camera device available at /dev/videoX (e.g., /dev/video0)

Installation

(Recommended) Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

Install dependencies

pip install -r requirements.txt

If pip is missing, install it first (device dependent), then retry.

Usage

Run real-time prediction using the INT8 TFLite model:

python3 stream_cam_predict.py --dev 0 --port 5000 --model models/mlp48_int8.tflite --flip 1

Arguments

--dev: Camera device index (0 = default camera, 1 = external camera, etc.)

--port: Port number used by the script (only relevant if the script serves a UI or stream)

--model: Path to the TFLite model file

--flip: Flip image horizontally (1 = enable, 0 = disable). Use flip=1 for “mirror view”.

Examples

Default camera, mirrored view:
python3 stream_cam_predict.py --dev 0 --model models/mlp48_int8.tflite --flip 1

External camera (index 1), no flip:
python3 stream_cam_predict.py --dev 1 --model models/mlp48_int8.tflite --flip 0

Troubleshooting

If the camera does not open: verify the device exists (e.g., /dev/video0) and that your user has permission.

If FPS is low: reduce camera resolution in the script (if supported) or ensure no other processes are using the camera.

If predictions are unstable: improve lighting, keep the hand fully visible, and hold the gesture steady for a short moment.