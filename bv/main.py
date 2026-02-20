from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import cv2
import os
import uuid
import subprocess
import tempfile

app = Flask(__name__)
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = os.path.join("static", "videos")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    video_file = None

    if request.method == "POST":
        file = request.files.get("video")

        if file and file.filename:
            input_filename = f"{uuid.uuid4().hex}_{file.filename}"
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            file.save(input_path)

            output_filename = f"{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)

            process_video(input_path, output_path)

            video_file = f"videos/{output_filename}"

    return render_template("index.html", video_file=video_file)


def process_video(input_path, output_path):
    frames_dir = "temp_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        frame_path = os.path.join(frames_dir, f"{frame_id:06d}.png")
        cv2.imwrite(frame_path, annotated)
        frame_id += 1

    cap.release()

    if frame_id == 0:
        raise RuntimeError("Видео не содержит кадров")

    ffmpeg_path = "/usr/bin/ffmpeg"

    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(int(fps)),
        "-i", f"{frames_dir}/%06d.png",
        "-c:v", "mpeg4",
        "-q:v", "5",
        output_path
    ]

    subprocess.run(cmd, check=True)

    shutil.rmtree(frames_dir)



if __name__ == "__main__":
    app.run(debug=True)
