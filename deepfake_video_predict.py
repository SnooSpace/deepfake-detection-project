"""
Deepfake Video Detection — End-to-End Inference
=================================================
Takes a raw video file (any format OpenCV can read: .mp4, .mov, .avi, etc.),
extracts frames, detects + crops the face in each frame (MTCNN), samples 24
frames matching the training pipeline, and runs them through the trained
CNN-LSTM model to produce a REAL / FAKE prediction with a confidence score.

Usage:
    python deepfake_video_predict.py --video path/to/video.mp4 --model path/to/FINAL_model.pt

Setup (one-time):
    pip install torch torchvision facenet-pytorch opencv-python pillow numpy

Notes:
- Works on CPU (slower, ~10-30s per video depending on length/hardware) or GPU if available.
- Face detection uses MTCNN (facenet-pytorch), matching the detector family used
  to build the training dataset (FaceForensics++), for consistent preprocessing.
- If MTCNN fails to detect a face in a given frame, that frame is skipped;
  if too few faces are found overall, the script warns rather than silently
  producing a low-quality prediction.
"""

import argparse
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Missing dependency. Install with: pip install facenet-pytorch")
    sys.exit(1)


# ============================================================
# Model definition — must match training exactly
# ============================================================
class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=1, unfreeze_last_n_blocks=4):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)  # weights loaded from checkpoint, not ImageNet, at inference time
        self.cnn = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1280
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = self.pool(features)
        features = features.view(batch_size, seq_len, self.feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_hidden = h_n[-1]
        logit = self.classifier(final_hidden)
        return logit.squeeze(1)


# ============================================================
# Preprocessing — must match training exactly
# ============================================================
SEQ_LEN = 24
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.30  # matches the final v3 operating point from training

transform_eval = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_frames(video_path, max_raw_frames=150):
    """Pull raw frames from the video file, evenly spaced, capped at max_raw_frames
    so very long videos don't blow up memory/time before face detection even runs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Video has no readable frames (corrupt file or unsupported codec).")

    n_to_read = min(total_frames, max_raw_frames)
    indices = np.linspace(0, total_frames - 1, n_to_read).astype(int)

    frames = []
    idx_set = set(indices.tolist())
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in idx_set:
            # BGR (OpenCV) -> RGB (PIL/torch convention)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("Failed to extract any frames from the video.")
    return frames


def detect_and_crop_faces(frames, detector, margin=0.25):
    """Run MTCNN on each frame, return cropped+aligned face images (PIL).
    Frames with no detected face are skipped."""
    face_crops = []
    for frame in frames:
        img = Image.fromarray(frame)
        boxes, probs = detector.detect(img)
        if boxes is None or len(boxes) == 0:
            continue

        # Use the highest-confidence detected face
        best_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_idx]

        # Add a margin around the tight box so the crop isn't overly tight (matches
        # typical FaceForensics++-style face-crop conventions)
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - margin * w)
        y1 = max(0, y1 - margin * h)
        x2 = min(img.width, x2 + margin * w)
        y2 = min(img.height, y2 + margin * h)

        crop = img.crop((x1, y1, x2, y2))
        face_crops.append(crop)

    return face_crops


def sample_sequence(face_crops, seq_len=SEQ_LEN):
    """Match the training-time sampling strategy: evenly spaced if enough frames,
    cyclic repeat-padding if too few."""
    n = len(face_crops)
    if n >= seq_len:
        indices = np.linspace(0, n - 1, seq_len).astype(int)
    else:
        indices = np.array([i % n for i in range(seq_len)])
    return [face_crops[i] for i in indices]


def predict(video_path, model_path, threshold=DEFAULT_THRESHOLD, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = DeepfakeCNNLSTM(unfreeze_last_n_blocks=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Loading face detector (MTCNN)...")
    detector = MTCNN(keep_all=True, device=device)

    print(f"Extracting frames from: {video_path}")
    raw_frames = extract_frames(video_path)
    print(f"  -> {len(raw_frames)} raw frames extracted")

    print("Detecting and cropping faces...")
    face_crops = detect_and_crop_faces(raw_frames, detector)
    print(f"  -> {len(face_crops)} frames with a detected face")

    if len(face_crops) == 0:
        print("\nERROR: No faces detected in this video. Cannot make a prediction.")
        print("Check that the video contains a visible, front-facing person.")
        sys.exit(1)
    if len(face_crops) < 8:
        print(f"\nWARNING: Only {len(face_crops)} face frames detected — prediction may be unreliable.")
        print("Best results come from clear, well-lit, front-facing video of at least a few seconds.")

    sampled = sample_sequence(face_crops, SEQ_LEN)
    tensors = [transform_eval(img) for img in sampled]
    sequence_tensor = torch.stack(tensors).unsqueeze(0).to(device)  # (1, seq_len, C, H, W)

    print("Running prediction...")
    with torch.no_grad():
        logit = model(sequence_tensor)
        prob_fake = torch.sigmoid(logit).item()

    label = "FAKE" if prob_fake > threshold else "REAL"

    print("\n" + "=" * 40)
    print(f"  Prediction: {label}")
    print(f"  FAKE probability: {prob_fake:.4f}")
    print(f"  Decision threshold: {threshold}")
    print("=" * 40)

    return {"label": label, "fake_probability": prob_fake, "threshold": threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection on a raw video file.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"Decision threshold for FAKE classification (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    predict(args.video, args.model, threshold=args.threshold)