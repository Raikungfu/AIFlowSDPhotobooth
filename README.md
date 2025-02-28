# AIFlowSDPhotobooth

AIFlowSDPhotobooth is an AI-powered image processing application that allows users to remove backgrounds, align faces, and apply inpainting techniques using Stable Diffusion models.

## Features
- Face alignment and background removal
- Image processing with AI-based inpainting
- Integration with Firebase for image storage
- Web API built with Flask
- Remote access via PageKite

## Requirements
Ensure you have the following installed before running the application:
- Python 3.8+
- CUDA-compatible GPU with drivers installed
- PyTorch with CUDA support
- Required Python dependencies (see `requirements.txt`)

## Installation
Clone the repository and install dependencies:

```sh
$ git clone https://github.com/Raikungfu/AIFlowSDPhotobooth.git
$ cd AIFlowSDPhotobooth
$ pip install -r requirements.txt
```

## Usage
### 1. Set Up Firebase
- Add your Firebase credentials to `firebase/social-networking-8c187-firebase-adminsdk-cbnj3-5738be94d4.json`.
- Update `firebaseConfig` in `app.py` with your Firebase storage bucket.

### 2. Run the Application
Start the Flask server:
```sh
$ python app.py
```
The API will run on `http://0.0.0.0:5000`.

### 3. Expose Server via PageKite (Optional)
If you need remote access, run:
```sh
$ python3 pagekite.py 5000 raiyugi.pagekite.me
```

## API Endpoints
### Process Image
**Endpoint:** `POST /process_image`

**Request:**
- `file`: Image file (PNG/JPG)
- `options`: JSON object with processing options
- `personal_info`: JSON object with additional parameters

**Response:**
```json
{
  "imageUrl": [
    "url_of_processed_image_1",
    "url_of_processed_image_2"
  ]
}
```

## Troubleshooting
- If encountering CUDA errors, ensure your GPU drivers and PyTorch installation support CUDA.
- Check Firebase configuration for authentication issues.
- If Stable Diffusion models are missing, manually download them from Hugging Face.

## License
This project is licensed under the MIT License.

## Author
To Bich Phuong