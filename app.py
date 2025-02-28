from flask import Flask, request, jsonify
import os
import json
from flask_cors import CORS
from threading import Thread
from enums.prompt_type import PromptType
from enums.user_info import Gender
from utils.background_prcessing import place_on_new_background, remove_background
from utils.face_alignment import align_face
from utils.firebase_utils import (
    initialize_firebase
)
from utils.image_upload import refine_with_depth2img_image, upload_image, inpaint_image
from PIL import Image

from utils.masks import generate_face_lower_mask


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/process_image', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    image_file = request.files['file']
    options = json.loads(request.form.get('options', '{}'))
    form_data = json.loads(request.form.get('personal_info', None))

    image = Image.open(image_file).convert("RGBA")
    aligned_image = align_face(image)

    image_outputs = []
    image_outputs.append(upload_image(aligned_image, bucket))

    if options.get("remove_background", False):
        image, image_mask = remove_background(aligned_image)
        image_outputs.append(upload_image(image_mask, bucket))

    new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = place_on_new_background(image, new_background).convert("RGB")

    mask_face, mask_clothes, mask_combined = generate_face_lower_mask(image, image_mask)
    if mask_face is None or mask_clothes is None:
        return jsonify({'error': 'No face detected, create mask failed'}), 400

    image_outputs.append(upload_image(mask_clothes, bucket))
    image_outputs.append(upload_image(mask_face, bucket))

    inpainted_image1 = inpaint_image(image, mask_face, form_data['seek'], PromptType.FACE, Gender(form_data['gender'].lower()))
    if inpainted_image1 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500

    image_outputs.append(upload_image(inpainted_image1, bucket))

    inpainted_image2 = inpaint_image(image, mask_clothes, form_data['seek'], PromptType.CLOTHES, Gender(form_data['gender'].lower()))
    if inpainted_image2 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500

    image_outputs.append(upload_image(inpainted_image2, bucket))

    inpainted_image3 = inpaint_image(image, mask_combined, form_data['seek'], PromptType.COMBINE, Gender(form_data['gender'].lower()))
    if inpainted_image3 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500
    
    image_outputs.append(upload_image(inpainted_image3, bucket))
    
    depth2img_image1 = refine_with_depth2img_image(inpainted_image1, form_data['seek'], PromptType.FACE, Gender(form_data['gender'].lower()))
    if depth2img_image1 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500
    
    image_outputs.append(upload_image(depth2img_image1, bucket))
    
    depth2img_image2 = refine_with_depth2img_image(inpainted_image2, form_data['seek'], PromptType.CLOTHES, Gender(form_data['gender'].lower()))
    if depth2img_image2 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500

    image_outputs.append(upload_image(depth2img_image2, bucket))

    
    depth2img_image3 = refine_with_depth2img_image(inpainted_image3, form_data['seek'], PromptType.COMBINE, Gender(form_data['gender'].lower()))
    if depth2img_image3 is None:
        return jsonify({'error': 'Failed to inpaint image'}), 500

    image_outputs.append(upload_image(depth2img_image3, bucket))

    return jsonify({'imageUrl': image_outputs})


def start_pagekite():
    os.system("python3 pagekite.py 5000 raiyugi.pagekite.me")


if __name__ == '__main__':
    cred_path = 'firebase/social-networking-8c187-firebase-adminsdk-cbnj3-5738be94d4.json'
    firebaseConfig = {
        'storageBucket': "social-networking-8c187.appspot.com"
    }
    
    bucket = initialize_firebase(cred_path, firebaseConfig)

    Thread(target=start_pagekite).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
