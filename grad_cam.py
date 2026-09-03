"""
Grad-CAM for the Deepfake CNN-LSTM Model
==========================================
Produces a heatmap over each analyzed frame showing which regions of the
face most influenced the model's REAL/FAKE decision. Useful for checking
whether the model is keying in on plausible deepfake artifacts (blending
seams, jawline/hairline boundaries, texture inconsistencies) versus
something spurious (background, compression noise unrelated to the face).

How it works (brief):
    Grad-CAM normally hooks a CNN's last convolutional layer. Here, the
    CNN's last block IS that layer — we hook it to capture (a) the
    activations during the forward pass and (b) the gradients that flow
    back into it when we backpropagate from the final FAKE-probability
    logit. Averaging those gradients gives an importance weight per
    channel; a weighted sum of the activation channels (ReLU'd) gives a
    coarse "where did the model look" heatmap, per frame.

Usage:
    python grad_cam.py --video path/to/video.mp4 --model FINAL_model.pt --out heatmaps.png

Setup (one-time, same as the other scripts):
    pip install torch torchvision facenet-pytorch opencv-python pillow numpy matplotlib
"""

import argparse
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Missing dependency. Install with: pip install facenet-pytorch")
    sys.exit(1)


# ============================================================
# Model definition — identical to training/inference scripts
# ============================================================
class DeepfakeCNNLSTM(nn.Module):
    def __init__(self, lstm_hidden_size=256, lstm_layers=1, unfreeze_last_n_blocks=4):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)
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
        features = self.cnn(x)          # <-- Grad-CAM hooks this
        features = self.pool(features)
        features = features.view(batch_size, seq_len, self.feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_hidden = h_n[-1]
        logit = self.classifier(final_hidden)
        return logit.squeeze(1)


SEQ_LEN = 24
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

transform_eval = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


# ============================================================
# Reused pipeline: video -> frames -> face crops -> sequence
# ============================================================
def extract_frames(video_path, max_raw_frames=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Video has no readable frames.")
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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("Failed to extract any frames.")
    return frames


def detect_and_crop_faces(frames, detector, margin=0.25):
    face_crops = []
    for frame in frames:
        img = Image.fromarray(frame)
        boxes, probs = detector.detect(img)
        if boxes is None or len(boxes) == 0:
            continue
        best_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_idx]
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - margin * w)
        y1 = max(0, y1 - margin * h)
        x2 = min(img.width, x2 + margin * w)
        y2 = min(img.height, y2 + margin * h)
        face_crops.append(img.crop((x1, y1, x2, y2)))
    return face_crops


def sample_sequence(face_crops, seq_len=SEQ_LEN):
    n = len(face_crops)
    if n >= seq_len:
        indices = np.linspace(0, n - 1, seq_len).astype(int)
    else:
        indices = np.array([i % n for i in range(seq_len)])
    return [face_crops[i] for i in indices]


# ============================================================
# Grad-CAM core
# ============================================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.model.cnn.register_forward_hook(self._save_activations)
        self.model.cnn.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out  # (B*seq_len, C, H, W)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B*seq_len, C, H, W)

    def generate(self, sequence_tensor):
        """sequence_tensor: (1, seq_len, C, H, W), requires no pre-set requires_grad."""
        self.model.zero_grad()
        logit = self.model(sequence_tensor)  # (1,)
        prob_fake = torch.sigmoid(logit)
        # Backprop from the logit (equivalently: pushing the FAKE-probability up)
        logit.backward()

        # activations/gradients shape: (seq_len, C, H, W)  [batch=1 was folded into B*seq_len]
        acts = self.activations.detach()
        grads = self.gradients.detach()

        # Channel importance weights = global-average-pooled gradients, per frame
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (seq_len, C, 1, 1)
        cams = F.relu((weights * acts).sum(dim=1))       # (seq_len, H, W)

        # Normalize each frame's CAM independently to [0, 1]
        cams_norm = []
        for i in range(cams.shape[0]):
            cam = cams[i]
            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min).item() > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            cams_norm.append(cam.cpu().numpy())

        return cams_norm, logit.item(), prob_fake.item()


def overlay_heatmap(pil_image, cam, alpha=0.45):
    """pil_image: PIL RGB image at IMG_SIZE x IMG_SIZE. cam: (h, w) array in [0,1]."""
    img_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlaid = (1 - alpha) * img_np + alpha * heatmap
    return np.clip(overlaid, 0, 1)


def run(video_path, model_path, out_path, num_frames_to_show=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = DeepfakeCNNLSTM(unfreeze_last_n_blocks=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # eval mode (BatchNorm/Dropout frozen) — gradients still flow fine for Grad-CAM

    gradcam = GradCAM(model)

    print("Loading face detector (MTCNN)...")
    detector = MTCNN(keep_all=True, device=device)

    print(f"Extracting frames from: {video_path}")
    raw_frames = extract_frames(video_path)
    print(f"  -> {len(raw_frames)} raw frames extracted")

    print("Detecting and cropping faces...")
    face_crops = detect_and_crop_faces(raw_frames, detector)
    print(f"  -> {len(face_crops)} frames with a detected face")
    if len(face_crops) == 0:
        print("ERROR: No faces detected.")
        sys.exit(1)

    sampled = sample_sequence(face_crops, SEQ_LEN)
    tensors = [transform_eval(img) for img in sampled]
    sequence_tensor = torch.stack(tensors).unsqueeze(0).to(device)
    sequence_tensor.requires_grad_(True)

    print("Running forward + backward pass for Grad-CAM...")
    cams, logit_val, prob_fake = gradcam.generate(sequence_tensor)

    label = "FAKE" if prob_fake > 0.30 else "REAL"
    print(f"\nPrediction: {label} (FAKE probability: {prob_fake:.4f})")

    # Build a grid of a subset of frames with heatmap overlays
    show_indices = np.linspace(0, len(sampled) - 1, min(num_frames_to_show, len(sampled))).astype(int)
    n = len(show_indices)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_idx, frame_idx in enumerate(show_indices):
        overlaid = overlay_heatmap(sampled[frame_idx], cams[frame_idx])
        axes[ax_idx].imshow(overlaid)
        axes[ax_idx].set_title(f"frame {frame_idx}", fontsize=9)
        axes[ax_idx].axis("off")
    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(f"Grad-CAM — prediction: {label} (p_fake={prob_fake:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap grid to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for the deepfake CNN-LSTM model.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--out", default="gradcam_output.png", help="Output image path")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to show in the grid")
    args = parser.parse_args()

    run(args.video, args.model, args.out, num_frames_to_show=args.num_frames)
