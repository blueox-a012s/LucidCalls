# Aegis Echo: Edge-AI Cantonese Scam Defense

Real-time multimodal detection of AI deepfakes and scam scripts for Hong Kong elderly, running fully offline on the edge.

## System Prerequisites (Jetson Orin)
Because this project is optimized for the NVIDIA Jetson ARM architecture, you must install system dependencies first:
sudo apt update
sudo apt install -y ffmpeg libopenblas-dev
(FFmpeg is required for audio decoding).

## Installation
1. Create a virtual environment:
   python3 -m venv scam_env
   source scam_env/bin/activate

2. Install PyTorch (CPU version for Jetson stability):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

3. Install the rest of the requirements:
   pip install -r requirements.txt

## Running the Application
1. Start the backend server:
   python ws_server.py
2. The server will load the ONNX Deepfake model, Whisper ASR, and the Qwen LLM.
3. Open index.html in a web browser.
4. Click "Live Mic" or "Upload File" to see the real-time scam detection and LLM reasoning in action.

## Hardware Acceleration Benchmarks
While this repository runs the pipeline on CPU for out-of-the-box compatibility, our engineering pipeline compiled the 1D-CNN into a TensorRT FP16 engine, achieving an inference latency of 0.066 ms on the NVIDIA Jetson AGX Orin. The trained deepfake_detector.onnx model is included in this repository.