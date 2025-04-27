import numpy as np


def fill_bottom_by_stretch(image, stretch_height=100, segment_width=10):
    h, w = image.shape[:2]

    for start_x in range(0, w, segment_width):
        end_x = min(start_x + segment_width, w)
        
        last_non_white_row = None
        for y in range(h - 1, -1, -1):
            row = image[y, start_x:end_x]
            white_pixels = np.sum(row > 240)
            total_pixels = len(row)
            
            if white_pixels < total_pixels * 0.02:
                last_non_white_row = y
                break
        
        if last_non_white_row is None:
            continue

        if last_non_white_row > 0:
            prev_row = image[last_non_white_row - 1, start_x:end_x]
            color_diff = np.sum(np.abs(prev_row - image[last_non_white_row, start_x:end_x]))
            stretch_height = max(50, int(color_diff / 100)) + 20

        stretch_start = last_non_white_row + 1
        stretch_end = min(h, stretch_start + stretch_height)

        for i in range(stretch_start, stretch_end):
            image[i, start_x:end_x] = image[last_non_white_row, start_x:end_x]

        if stretch_end < h:
            image[stretch_end:, start_x:end_x] = image[last_non_white_row, start_x:end_x]

    return image