from PIL import Image
import io


def optimize_image(image, max_size=(1024, 1024), quality=95):
    image.thumbnail(max_size, Image.LANCZOS)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG2000", quality_mode="dB", quality_layers=[quality])
    
    output_buffer.seek(0)
    optimized_image = Image.open(output_buffer)

    return optimized_image
