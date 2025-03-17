import os

import requests
from flask import Flask, request, jsonify, send_from_directory, send_file

from functions import download_file, process_video

app = Flask(__name__)

# Define media storage path
# MEDIA_FOLDER = os.path.join(os.getcwd(), "media", "videos")
MEDIA_FOLDER = os.path.expanduser("~/media/videos")
# IMAGES_FOLDER = os.path.expanduser("~/video_builder/images")
os.makedirs(MEDIA_FOLDER, exist_ok=True)  # Ensure media folder exists

@app.route('/')
def hello_world():  # put application's code here
    return 'Welcome to Video Builder!'


@app.route('/process-video', methods=['POST'])
def process():
    data = request.get_json()

    # Load video
    output_path = os.path.join(MEDIA_FOLDER, f"{data.get('uid')}.mp4")  # Store in media folder

    vid_url = data.get('vid_url')
    if not vid_url:
        return jsonify({'error': 'Missing vid_url'})

    # video_path = os.path.join(MEDIA_FOLDER,"bg-change-final-video.mp4")
    video_path = download_file(vid_url)
    if not video_path:
        return jsonify({'error': 'Failed to download video'}), 400
    qr_code = download_file(data.get('qr_code'))
    logo1 = download_file(data.get('logo1'))
    logo2 = download_file(data.get('logo2'))
    logo3 = download_file(data.get('logo3'))
    logo1_txt = data.get('logo1_txt')
    logo2_txt = data.get('logo2_txt')
    logo3_txt = data.get('logo3_txt')
    top_img = f"~/video_builder/images/img.png"
    # qr_code = os.path.join(MEDIA_FOLDER,"qr_code.png")
    # logo1 = os.path.join(MEDIA_FOLDER,"logo")
    # logo2 = os.path.join(MEDIA_FOLDER,"logo3")
    # logo3 = os.path.join(MEDIA_FOLDER,"logo")
    # logo1_txt = 'logo1_txt'
    # logo2_txt = 'logo2_txt'
    # logo3_txt = 'logo3_txt'
    output_path = process_video(video_path, top_img, qr_code, logo1, logo1_txt, logo2, logo2_txt, logo3, logo3_txt, output_path)
    output_filename = os.path.basename(output_path)

    return jsonify({"download_url": f"/media/videos/{output_filename}"}), 200


@app.route('/media/videos/<filename>')
def serve_media(filename):
    """Serve processed videos from the media folder."""
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
