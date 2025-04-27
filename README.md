# AIFlowSDPhotobooth

📸 AIFlowSDPhotobooth is an AI Inference Server for generating high-quality ID photos and stylized images using Stable Diffusion 1.5 + Adapter + ControlNet (planned: FLUX.1-dev + controlnet union).

---

## 🌐 Features

- Receive image via message queue.
- Using StableDiffusionInpaintPipeline/StableDiffusionControlNetInpaintPipeline pipeline to modify photo.
- Detects face and preprocesses ID photos.
- Adjust size of photo match ratio 4\*6 for ID photos
- Optimizes and aligns face position.
- Removes background.
- Generates masks.
- Use ControlNet modules include OpenPose, Depth, Canny with Controlnet images are OpenPose, Scribble, Depth, Canny, Soft Edge for another photo style.
- Inpaints with/without Controlnets.
- Uploads processed image to Firebase Storage and returns list images processed to API server via queue.

---

## 🧱 Model Reference

```
models/ip-adapter/ip-adapter-plus-face_sd15.pth  # Source: [Tencent AI Lab - IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
models/dlib/shape_predictor_68_face_landmarks.dat  # Source: [dlib.net](http://dlib.net/files/)
botp/stable-diffusion-v1-5-inpainting  # Source: [RunwayML SD Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
lllyasviel/control_v11p_sd15_openpose  # Source: [lllyasviel/ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose)
lllyasviel/ControlNet  # Source: [lllyasviel/ControlNet Base](https://huggingface.co/lllyasviel/ControlNet)
lllyasviel/control_v11f1p_sd15_depth  # Source: [Depth ControlNet](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)
lllyasviel/control_v11p_sd15_canny  # Source: [Canny ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)
Intel/zoedepth-nyu-kitti  # Source: [ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth)
```

---

## 📆 Installation

### 1. Clone repository

```bash
git clone https://github.com/Raikungfu/AIFlowSDPhotobooth.git
cd AIFlowSDPhotobooth
```

### 2. Download required models

Install `gdown`:

```bash
pip install gdown
```

Download models:

```bash
# Dlib Shape Predictor (source: dlib.net)
gdown --id 1WdwqdrivpYtOxUVYKrR3hV9E_gYH0cu_ -O models/dlib/shape_predictor_68_face_landmarks.dat

# IP-Adapter (source: Tencent AI Lab)
gdown --id 1SDvNYNrlEj1oqfuRq5ScOLK4rXC_43aq -O models/ip-adapter/ip-adapter-plus-face_sd15.pth
```

### 📁 Directory Structure (for models)

```
models/
├── dlib/
│   └── shape_predictor_68_face_landmarks.dat
├── ip-adapter/
│   └── ip-adapter-plus-face_sd15.pth
└── (others handled via Hugging Face auto-download or torch hub)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run App Locally

```bash
python app.py
```

---

## 🐳 Docker Build & Run

Ensure Docker is installed.

### 1. Build Docker Image

```bash
docker build -t sd-photobooth .
```

### 2. Run Container

```bash
docker run -p 7860:7860 sd-photobooth
```

> Ensure that model files are mounted into the container or pre-included in the Docker build context.

---

## 🧪 Run on Google Colab

To run on Google Colab:

1. Upload the project code or clone repo via `!git clone ...`
2. Install dependencies:

```python
!pip install -r requirements.txt
```

3. Install additional system packages (if needed):

```bash
!apt update && apt install -y ffmpeg libgl1-mesa-glx
```

4. Download models using `gdown` (as above).
5. Run the app using:

```python
!python app.py
```

---

## 🧠 Flow Overview

```
Receive message(image url, photo style, correlation id) via message queue
     ↓
Check type: ID Photo?
     ↓
If Yes → Detect face → Validate?
     ↓        ↘
   Yes        No
   ↓           ↓
Rotate/Center  → Skip
     ↓
Resize Canvas
     ↓
Stretch Clothes
     ↓
Optimize Image
     ↓
Remove Background
     ↓
Generate mask
     ↓
Branch by Type:
    - ID Photo → Inpaint clothes → Inpaint all
    - OpenPose/Scribble/Depth/Canny → Generate control image → Inpaint
     ↓
Upload Final Image
     ↓
Send Result via message queue
```

---

## 👨‍💻 Author

**To Bich Phuong**

---

## 📄 License

This project is licensed under the **MIT License**.
