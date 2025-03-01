from flask import Flask, request, jsonify
import os
import json
from flask_cors import CORS
from threading import Thread
from enums.prompt_type import PromptType
from enums.user_info import Gender
from utils.background_prcessing import place_on_new_background, remove_background
from utils.face_alignment import align_face
from utils.firebase_utils import initialize_firebase
from utils.image_upload import upload_image, inpaint_image
from PIL import Image
from utils.masks import generate_face_lower_mask


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/process_image', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    try:
        image_file = request.files['file']
        options = json.loads(request.form.get('options', '{}'))
        form_data = json.loads(request.form.get('personal_info', '{}'))

        edit_strength = form_data.get('edit_strength', {'face': 1.0, 'clothes': 1.0, 'combine': 1.0})
        guidance_scale = form_data.get('guidance_scale', {'face': 7.5, 'clothes': 7.5, 'combine': 7.5})

        seek_value = form_data.get('seek', 0)
        gender = Gender(form_data.get('gender', 'male').lower())

        image = Image.open(image_file).convert("RGBA")
        aligned_image = align_face(image)
        image_outputs = [upload_image(aligned_image, bucket)]

        if options.get("remove_background"):
            image, image_mask = remove_background(aligned_image)
            image_outputs.append(upload_image(image_mask, bucket))
        else:
            image_mask = None

        if image_mask:
            new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = place_on_new_background(image, new_background).convert("RGB")

        mask_face, mask_clothes, mask_combined = generate_face_lower_mask(image, image_mask)
        if not mask_face or not mask_clothes:
            return jsonify({'error': 'No face detected, create mask failed'}), 400

        image_outputs.extend([
            upload_image(mask_clothes, bucket),
            upload_image(mask_face, bucket)
        ])

        edit_types = {
            PromptType.FACE: (mask_face, edit_strength.get('face', 1.0), guidance_scale.get('face', 7.5)),
            PromptType.CLOTHES: (mask_clothes, edit_strength.get('clothes', 1.0), guidance_scale.get('clothes', 7.5)),
            PromptType.COMBINE: (mask_combined, edit_strength.get('combine', 1.0), guidance_scale.get('combine', 7.5))
        }

        for edit_type, (mask, strength, scale) in edit_types.items():
            inpainted_image = inpaint_image(image, mask, seek_value, edit_type, gender, strength, scale)
            if inpainted_image:
                image_outputs.append(upload_image(inpainted_image, bucket))
            else:
                return jsonify({'error': f'Failed to inpaint {edit_type.name.lower()}'}), 500

        return jsonify({'imageUrl': image_outputs})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



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
