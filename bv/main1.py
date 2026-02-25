from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join("static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

TEMP_MIN = 0.0
TEMP_MAX = 100.0
DELTA_T = 5.0
MIN_AREA = 50
BLUR_KERNEL = 31


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return "No file", 400

        input_filename = f"{uuid.uuid4().hex}_{file.filename}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)

        output_filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        process_image(input_path, output_path)

        return f"/static/results/{output_filename}"

    return render_template("index1.html")


def process_image(input_path, output_path):

    frame = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    temp_map = TEMP_MIN + (frame / 255.0) * (TEMP_MAX - TEMP_MIN)
    temp_map = temp_map.astype(np.float32)

    blur = cv2.GaussianBlur(temp_map, (BLUR_KERNEL, BLUR_KERNEL), 0)
    diff = temp_map - blur

    mask = (diff > DELTA_T).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    output = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue

        center_x, center_y = centroids[i]
        cv2.circle(output, (int(center_x), int(center_y)), 6, (0, 0, 255), -1)

    cv2.imwrite(output_path, output)


if __name__ == "__main__":
    app.run(debug=True)
