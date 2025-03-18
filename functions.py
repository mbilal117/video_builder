import os
import cv2
import tempfile
import requests
import numpy as np


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

        img_roi = img[y:y+h, x:x+w]  # Extract corresponding region from the base image
        img[y:y+h, x:x+w] = (1 - alpha_mask[:, :, None]) * img_roi + (alpha_mask[:, :, None]) * overlay_img

        return img
    except Exception as e:
        print(f"Error overlaying image: {e}")
        return img  # Return original image in case of failure


def add_text(frame, text, position, font_scale=0.3, shadow_offset=0, thickness=1, color=(255, 255, 255)):
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

def process_video(bkg_vid_path, top_img, qr_code, logo1, logo1_txt, logo2, logo2_txt, logo3, logo3_txt, output_path, property_adrs, zip_code, city, state, county):

    import os
    cwd = os.getcwd()
    try:
        # Validate video file
        if not os.path.exists(bkg_vid_path):
            raise FileNotFoundError(f"Background video not found: {bkg_vid_path}")

        cap = cv2.VideoCapture(bkg_vid_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Load logos (ensure they have alpha channels)
        # upper_image = cv2.imread(bkg_img, cv2.IMREAD_UNCHANGED)
        qr_code = cv2.imread(qr_code, cv2.IMREAD_UNCHANGED)
        logo1 = cv2.imread(logo1, cv2.IMREAD_UNCHANGED)
        logo2 = cv2.imread(logo2, cv2.IMREAD_UNCHANGED)
        logo3 = cv2.imread(logo3, cv2.IMREAD_UNCHANGED)
        upper_image = cv2.imread(f"{cwd}/images/img.png", cv2.IMREAD_UNCHANGED)
        # Define the text to display above each logo
        texts = [logo1_txt, logo2_txt, logo3_txt]
        address = [property_adrs, f"{city}, {state} {zip_code}", county]


        # Resize upper image to fit inside the frame with 25% margins
        upper_width = int(frame_width * 0.821)  # 50% of frame width
        upper_height = int(frame_height * 0.40)  # 50% of frame height
        upper_image = resize_overlay(upper_image, upper_width, upper_height)

        # Center the upper image inside the frame (25% margin on each side)
        upper_x = (frame_width - upper_width) // 2  # Center horizontally
        upper_y = int(frame_height * 0.055)  # 25% from the top

        # Resize QR code
        qr_code = resize_overlay(qr_code, frame_width // 10, frame_height // 10)  # Small QR Code

        # Resize logos
        logo_height = frame_height // 10  # 10% of video height
        logo_width = frame_width // 4  # Each logo should be 1/6th of video width

        logo1_resized = resize_overlay(logo1, logo_width, logo_height)
        logo2_resized = resize_overlay(logo2, logo_width, logo_height)
        logo3_resized = resize_overlay(logo3, logo_width, logo_height)

        # Define available space for 3 columns (increased width and height)
        allocated_width = int(frame_width * 0.6)  # Use 60% of the frame width for logos (increased)
        gap = 10  # Space between logos

        # Calculate total spacing and adjust column width accordingly
        total_spacing = gap * 2  # Two gaps between three logos
        column_width = (allocated_width - total_spacing) // 3  # Adjust column width

        # Increase height of the row
        row_height = int(logo_height * 0.5)  # Increase row height by 1.5 times

        # Define the starting position (closer to the left side)
        start_x = 35  # Offset from the left edge
        bottom_y = frame_height - row_height - 20  # Move row higher and adjust height

        ## Determine the merging point (Bottom of Upper Image)
        qr_y = upper_y + upper_height - 20  # Bottom edge of the upper image
        qr_x = upper_x + 5 # Center horizontally
        address_y = qr_y + 120  # Below QR code

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_color = (255, 255, 255)  # White color
        thickness = 1

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends

            # Overlay upper image at adjusted position
            frame = overlay_image_alpha(frame, upper_image, upper_x, upper_y)

            # Overlay QR Code
            frame = overlay_image_alpha(frame, qr_code, qr_x, qr_y)

            # Add Address Below QR Code
            frame = add_text(frame, property_adrs, (qr_x, address_y), 0.7, 1, 2)
            # frame = cv2.putText(frame, property_adrs, (qr_x, address_y ), font, 0.7, font_color, thickness+1, cv2.LINE_AA)
            for i, address_line in enumerate(address[1:]):
                frame = add_text(frame, address_line, (qr_x, address_y +20+ (i * 10)), 0.35)
                # frame = cv2.putText(frame, address_line, (qr_x, address_y +20+ (i * 10)), font, font_scale + 0.05, font_color, thickness, cv2.LINE_AA)

            # Overlay logos
            logos = [logo1_resized, logo2_resized, logo3_resized]
            # Place logos in the 3 columns (on the left side)
            for i, logo in enumerate(logos[:3]):  # Ensure only 3 logos are placed
                # Resize logo to fit inside column width while maintaining aspect ratio
                scale_factor = column_width / logo.shape[1]  # Scale based on width
                new_height = int(logo.shape[0] * scale_factor)  # Maintain aspect ratio
                resized_logo = cv2.resize(logo, (column_width, new_height), interpolation=cv2.INTER_AREA)

                # Compute x position for upper-left corner placement
                logo_x = start_x + (i * (column_width + gap))  # No centering, aligns to the left of each cell
                logo_y = bottom_y  # Logos are positioned on the bottom of the row

                # Overlay resized logo on the frame
                frame = overlay_image_alpha(frame, resized_logo, logo_x, logo_y)

                # Position for left-aligned text above each logo
                text_x = logo_x  # Left-aligned with the logo
                text_y = logo_y - 10  # Text is 10px above the logo

                # Overlay text above each logo
                frame = add_text(frame, texts[i], (text_x, text_y))
                # frame = cv2.putText(frame, texts[i], (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

            out.write(frame)  # Write modified frame to output video

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return output_path
    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
        return None
    except ValueError as val_error:
        print(f"Value error: {val_error}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
