import os
import cv2
import tempfile
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cwd = os.getcwd()
fonts_dir = os.path.join(cwd, 'fonts')
font_path_inter_bold = os.path.join(fonts_dir, 'Inter_24pt-Black.ttf')
font_path_plus_jakarta = os.path.join(fonts_dir, 'PlusJakartaSans-Bold.ttf')

def download_file(url):
    """Download file from url and save it temporarily"""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[-1])
    with open(temp_file.name, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    return temp_file.name



def resize_overlay(overlay, target_width, target_height):
    """Resizes the overlay while maintaining aspect ratio."""
    try:
        if overlay is None:
            raise ValueError("Image is None, cannot resize.")
        h, w = overlay.shape[:2]
        aspect_ratio = w / h

        if w > h:  # Wider logo
            new_w = target_width
            new_h = int(target_width / aspect_ratio)
        else:  # Taller or square logo
            new_h = target_height
            new_w = int(target_height * aspect_ratio)

        return cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # return cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None  # Return None if resizing fails


def overlay_image_alpha(img, overlay, x, y):
    """Overlays `overlay` onto `img` at (x, y), preserving transparency."""
    try:
        if img is None or overlay is None:
            raise ValueError("One or more images are None.")

        if overlay.shape[2] == 3:  # Ensure overlay has an alpha channel
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        h, w = overlay.shape[:2]

        # Ensure overlay does not exceed image boundaries
        if y + h > img.shape[0] or x + w > img.shape[1]:
            return img

        overlay_img = overlay[:, :, :3]  # Extract RGB
        alpha_mask = overlay[:, :, 3] / 255.0  # Normalize alpha channel

        img_roi = img[y:y + h, x:x + w]  # Extract corresponding region from the base image
        img[y:y + h, x:x + w] = (1 - alpha_mask[:, :, None]) * img_roi + (alpha_mask[:, :, None]) * overlay_img

        return img
    except Exception as e:
        print(f"Error overlaying image: {e}")
        return img  # Return original image in case of failure


def add_text(frame, text, position, font_scale=0.8, shadow_offset=0, thickness=1, color=(255, 255, 255)):
    """Adds clear, high-quality text to a frame."""
    try:
        if frame is None:
            raise ValueError("Frame is None, cannot add text.")

        font = cv2.FONT_HERSHEY_SIMPLEX  # Try COMPLEX or TRIPLEX for better clarity

        # Add a shadow for better readability
        shadow_color = (25, 25, 25)  # Black shadow
        cv2.putText(frame, text, (position[0] + shadow_offset, position[1] + shadow_offset), font,
                    font_scale, shadow_color, thickness + 1, cv2.LINE_AA)

        # Add the main text
        return cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    except Exception as e:
        print(f"Error adding text: {e}")
        return frame  # Return original frame if text addition fails


def draw_texts_on_frame(frame, text, position, color, font_family, font_size):
    # Convert OpenCV frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Load font
    font = ImageFont.truetype(font_family, font_size)  # You can adjust the font size here
    current_position = position
    for char in text:
        draw.text(current_position, char, font=font, fill=color)
        # Adjust the position for the next character (letter-spacing = 0%)
        current_position = (current_position[0] + font.getbbox(char)[2], current_position[1])

    # Convert PIL image back to OpenCV frame
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def process_video(params):
    try:
        if not os.path.exists(params["bkg_vid_path"]):
            raise FileNotFoundError(f"Background video not found: {params['bkg_vid_path']}")

        cap = cv2.VideoCapture(params["bkg_vid_path"])
        if not cap.isOpened():
            print("Error: Unable to open the video file.")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(30, int(cap.get(cv2.CAP_PROP_FPS)))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(params["output_path"], fourcc, fps, (frame_width, frame_height))

        upper_image = cv2.imread(f"{cwd}/images/img.png", cv2.IMREAD_UNCHANGED)
        # upper_image = cv2.imread(params["top_img"], cv2.IMREAD_UNCHANGED)
        qr_code = cv2.imread(params["qr_code"], cv2.IMREAD_UNCHANGED)
        logo1 = cv2.imread(params["logo1"], cv2.IMREAD_UNCHANGED)
        logo2 = cv2.imread(params["logo2"], cv2.IMREAD_UNCHANGED)
        logo3 = cv2.imread(params["logo3"], cv2.IMREAD_UNCHANGED)

        texts = [params["logo1_txt"], params["logo2_txt"], params["logo3_txt"]]
        address = [params["property_adrs"], f"{params['city']}, {params['state']} {params['zip_code']}", params['county']]

        upper_width = int(frame_width * 0.80)
        upper_height = int(frame_height * 0.40)
        upper_image = resize_overlay(upper_image, upper_width, upper_height)

        upper_x = (frame_width - upper_width) // 2
        upper_y = int(frame_height * 0.060)

        qr_code_size = min(frame_width // 6, frame_height // 6)
        qr_code = resize_overlay(qr_code, qr_code_size, qr_code_size)

        logo_height = frame_height // 15
        logo_width = frame_width // 4
        logo1_resized = resize_overlay(logo1, logo_width, logo_height)
        logo2_resized = resize_overlay(logo2, logo_width, logo_height)
        logo3_resized = resize_overlay(logo3, logo_width, logo_height)

        allocated_width = int(frame_width * 0.6)
        gap = 10
        total_spacing = gap * 2
        column_width = (allocated_width - total_spacing) // 3
        row_height = int(logo_height * 1.40)

        start_x = upper_x
        bottom_y = frame_height - row_height - 5

        ## ✅ **QR Code Left-Aligned**
        qr_x = upper_x + 15  # 15px from the left edge of the upper image
        # Get the real bottom Y of the upper image
        upper_bottom_y = upper_y + upper_image.shape[0]
        # qr_y = upper_y + upper_height - (qr_code_size // 2)
        qr_y = upper_y + upper_image.shape[0] - (qr_code_size // 2)
        address_y = qr_y + 270

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = overlay_image_alpha(frame, upper_image, upper_x, upper_y)
            frame = overlay_image_alpha(frame, qr_code, qr_x, qr_y)

            frame = draw_texts_on_frame(frame, params["property_adrs"], (qr_x, address_y), (255, 255, 255), font_path_inter_bold, 70)

            for i, address_line in enumerate(address[1:]):
                frame = draw_texts_on_frame(frame, address_line, (qr_x, address_y + 100 + (i * 50)),
                                            (255, 255, 255), font_path_plus_jakarta, 32)

            logos = [logo1_resized, logo2_resized, logo3_resized]
            for i, logo in enumerate(logos[:3]):
                scale_factor = column_width / logo.shape[1]
                new_height = int(logo.shape[0] * scale_factor)
                resized_logo = cv2.resize(logo, (column_width, new_height), interpolation=cv2.INTER_CUBIC)

                logo_x = start_x + (i * (column_width + gap))
                logo_y = bottom_y

                frame = overlay_image_alpha(frame, resized_logo, logo_x, logo_y)

                text_x = logo_x
                text_y = logo_y - 40
                frame = draw_texts_on_frame(frame, texts[i], (text_x, text_y), (255, 255, 255),
                                            'PlusJakartaSans-VariableFont_wght.ttf', 20)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return params["output_path"]
    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
        return None
    except ValueError as val_error:
        print(f"Value error: {val_error}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
