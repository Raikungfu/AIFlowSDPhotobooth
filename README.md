# AIFlowSDPhotobooth

📸 AIFlowSDPhotobooth is an AI Inference Server for generating high-quality ID photos and stylized images using Stable Diffusion 1.5 + Adapter + ControlNet (planned: FLUX.1-dev).

## 🌐 Features
- Detects face and preprocesses ID photos.
- Optimizes and aligns face position.
- Removes background.
- Generates masks, clothes, and session info.
- Supports ControlNet modules: OpenPose, Scribble, Depth, Canny, Soft Edge.
- Inpaints clothes or entire photo.
- Uploads processed image and returns session info.

## 🧱 Model Structure
```python
MODEL_PATHS = {
    "ip_adapter": "models/ip-adapter/ip-adapter-plus-face_sd15.pth",
    "shape_predictor": "models/dlib/shape_predictor_68_face_landmarks.dat",
    "base_model": "black-forest-labs/FLUX.1-dev",
    "controlnet_union": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "inpainting_model": "botp/stable-diffusion-v1-5-inpainting",
    "controlnet_openpose": "lllyasviel/control_v11p_sd15_openpose",
    "openpose_detection": "lllyasviel/ControlNet",
    "controlnet_depth": "lllyasviel/control_v11f1p_sd15_depth",
    "controlnet_canny": "lllyasviel/control_v11p_sd15_canny",
    "midas": "intel-isl/MiDaS",
}
📦 Installation
Clone repository

bash
Copy
Edit
git clone https://github.com/Raikungfu/AIFlowSDPhotobooth.git
cd AIFlowSDPhotobooth
Download required models
Install gdown for downloading models.

bash
Copy
Edit
pip install gdown
Then, download the necessary models:

bash
Copy
Edit
# Dlib Shape Predictor
gdown --id 1WdwqdrivpYtOxUVYKrR3hV9E_gYH0cu_ -O models/dlib/shape_predictor_68_face_landmarks.dat

# IP-Adapter
gdown --id 1SDvNYNrlEj1oqfuRq5ScOLK4rXC_43aq -O models/ip-adapter/ip-adapter-plus-face_sd15.pth
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
🚀 Run App
To start the app:

bash
Copy
Edit
python app.py
🧠 Flow Overview
mermaid
Copy
Edit
graph TD;
    A[Receive message] --> B{ID Photo?}
    B -->|Yes| C[Detect face]
    C --> D{Valid?}
    D -->|Yes| E[Rotate/Center face]
    E --> F[Resize canvas proportionally]
    F --> G[Stretch clothes to fill]
    G --> H[Optimize image]
    D -->|No| H
    B -->|Others| H
    H --> I[Background remover]
    I --> J[Generate mask, clothes, sex, combine]
    J --> K{Type}
    K -->|ID Photo| L[Inpaint clothes]
    L --> M[Inpaint All]
    K -->|OpenPose| N[Create OpenPose image]
    N --> O[Create Canny image]
    K -->|Scribble| P[Create scribble image]
    K -->|Depth| Q[Create Depth image]
    K -->|Soft Edge| R[Create Soft Edge image]
    O & P & Q & R --> S[Inpaint with ControlNets]
    M --> T[Upload Image]
    S --> T
    T --> U[Send result: URL, metadata, session code]

## License
This project is licensed under the MIT License.

## Author
To Bich Phuong