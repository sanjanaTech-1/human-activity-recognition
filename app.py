import os
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision.models.video import r3d_18


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_3d_resnet_ucf101.pth"
)

NUM_FRAMES = 16
IMAGE_SIZE = 112

TOP_K = 5


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

CLASS_NAMES = checkpoint["classes"]

NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)


model = r3d_18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")
print("Using device:", DEVICE)


# ============================================================
# VIDEO PREPROCESSING
# ============================================================

def preprocess_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise ValueError(
            "Could not open video"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30

    duration = (
        total_frames / fps
    )

    if total_frames <= 0:

        cap.release()

        raise ValueError(
            "Video contains no readable frames"
        )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    frames = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:

            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        frames.append(
            frame
        )

    cap.release()

    if len(frames) == 0:

        raise ValueError(
            "Could not extract frames"
        )

    while len(frames) < NUM_FRAMES:

        frames.append(
            frames[-1].copy()
        )

    frames = frames[
        :NUM_FRAMES
    ]

    video = np.stack(
        frames
    )

    video = torch.from_numpy(
        video
    ).float() / 255.0

    video = video.permute(
        0,
        3,
        1,
        2
    )

    video = video.permute(
        1,
        0,
        2,
        3
    )

    mean = torch.tensor(
        [
            0.43216,
            0.394666,
            0.37645
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1,
        1
    )

    std = torch.tensor(
        [
            0.22803,
            0.22145,
            0.216989
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1,
        1
    )

    video = (
        video - mean
    ) / std

    video = video.unsqueeze(
        0
    )

    return (
        video,
        duration,
        total_frames,
        fps
    )


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def get_reliability(confidence):

    if confidence >= 80:

        return "Very High"

    elif confidence >= 60:

        return "High"

    elif confidence >= 40:

        return "Moderate"

    elif confidence >= 20:

        return "Low"

    else:

        return "Very Low"


def create_analysis_summary(
    activity,
    confidence,
    duration
):

    if confidence >= 80:

        return (
            f"The model identified "
            f"{activity} with very high "
            f"confidence across the "
            f"analyzed video sequence."
        )

    elif confidence >= 60:

        return (
            f"The analyzed motion pattern "
            f"strongly corresponds to "
            f"{activity}."
        )

    elif confidence >= 40:

        return (
            f"The model detected motion "
            f"patterns associated with "
            f"{activity}, although the "
            f"prediction has moderate "
            f"certainty."
        )

    else:

        return (
            f"The model produced a low-confidence "
            f"prediction for {activity}. "
            f"The video may contain motion "
            f"patterns that overlap with other "
            f"trained activities."
        )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "success",

        "message":
        "Human Activity Recognition API is running",

        "model":
        "3D ResNet-18",

        "classes":
        CLASS_NAMES,

        "device":
        str(DEVICE)

    })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "video" not in request.files:

        return jsonify({

            "success": False,

            "error":
            "No video file uploaded"

        }), 400


    video_file = request.files[
        "video"
    ]


    if video_file.filename == "":

        return jsonify({

            "success": False,

            "error":
            "No video selected"

        }), 400


    temporary_path = None

    start_time = time.time()


    try:

        # ----------------------------------------------------
        # SAVE TEMPORARY VIDEO
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            video_file.save(
                temp_file.name
            )

            temporary_path = (
                temp_file.name
            )


        # ----------------------------------------------------
        # PREPROCESS VIDEO
        # ----------------------------------------------------

        video, duration, total_frames, fps = \
            preprocess_video(
                temporary_path
            )


        video = video.to(
            DEVICE
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                video
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


        # ----------------------------------------------------
        # TOP K PREDICTIONS
        # ----------------------------------------------------

        top_count = min(
            TOP_K,
            NUM_CLASSES
        )


        top_probabilities, top_indices = \
            torch.topk(
                probabilities,
                top_count,
                dim=1
            )


        top_predictions = []


        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        ):

            confidence = (
                probability.item()
                * 100
            )


            top_predictions.append({

                "activity":
                CLASS_NAMES[
                    index.item()
                ],

                "confidence":
                round(
                    confidence,
                    2
                )

            })


        # ----------------------------------------------------
        # PRIMARY PREDICTION
        # ----------------------------------------------------

        primary_activity = \
            top_predictions[0][
                "activity"
            ]


        primary_confidence = \
            top_predictions[0][
                "confidence"
            ]


        reliability = \
            get_reliability(
                primary_confidence
            )


        analysis_summary = \
            create_analysis_summary(
                primary_activity,
                primary_confidence,
                duration
            )


        processing_time = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # RETURN PROFESSIONAL ANALYSIS
        # ----------------------------------------------------

        return jsonify({

            "success": True,


            "result": {

                "activity":
                primary_activity,

                "confidence":
                primary_confidence,

                "reliability":
                reliability

            },


            "top_predictions":
            top_predictions,


            "video_analysis": {

                "duration_seconds":
                round(
                    duration,
                    2
                ),

                "total_frames":
                total_frames,

                "fps":
                round(
                    fps,
                    2
                ),

                "frames_analyzed":
                NUM_FRAMES

            },


            "model_info": {

                "architecture":
                "3D ResNet-18",

                "device":
                str(DEVICE),

                "classes":
                NUM_CLASSES

            },


            "analysis_summary":
            analysis_summary,


            "processing_time_seconds":
            round(
                processing_time,
                2
            )

        })


    except Exception as error:

        print(
            "Prediction Error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
            str(error)

        }), 500


    finally:

        if (

            temporary_path

            and

            os.path.exists(
                temporary_path
            )

        ):

            os.remove(
                temporary_path
            )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "Human Activity Recognition API"
    )

    print(
        "===================================="
    )

    print(
        "Server running on:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "====================================\n"
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )