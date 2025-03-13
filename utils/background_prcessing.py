from PIL import Image
from rembg import remove
import cv2
import numpy as np

def remove_background(image):
    try:
        transparent_image = remove(image, model_type="u2net_human_seg", alpha_matting=True,
                                   alpha_matting_foreground_threshold=240,
                                   alpha_matting_background_threshold=10,
                                   alpha_matting_erode_size=5)

        alpha_channel = transparent_image.split()[-1]
        mask = np.array(alpha_channel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filtered = np.zeros_like(mask)
        cv2.drawContours(mask_filtered, [largest_contour], -1, (255), thickness=cv2.FILLED)

        kernel = np.ones((5,5), np.uint8)
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel)
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel)

        img_cv = np.array(image)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (1, 1, img_cv.shape[1] - 2, img_cv.shape[0] - 2)
        mask_grab = np.where((mask_filtered > 128), cv2.GC_FGD, cv2.GC_BGD).astype('uint8')
        cv2.grabCut(img_cv, mask_grab, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        mask_final = np.where((mask_grab == 2) | (mask_grab == 0), 0, 255).astype('uint8')

        mask_final = Image.fromarray(mask_final)

        return transparent_image, mask_final
    except Exception as e:
        print(f"Error removing background: {e}")
        return None, None


def place_on_new_background(foreground, new_background):
    new_background = new_background.resize(foreground.size, Image.Resampling.LANCZOS).convert("RGBA")
    return Image.alpha_composite(new_background, foreground)