"""
Multimodal (video + audio) deepfake scoring. Needs, in the same folder:
  deepfake_video_predict.py   (your existing video pipeline - reused as-is)
  audio_branch.py             (audio model definition + constants)

CLI:
  python multimodal.py --video clip.mp4 --video-model FINAL_model.pt \
         --audio-model audio_run_fixed/audio_best.pt --audio-fix --floor 30
"""
import argparse, json, os, subprocess, tempfile, wave
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------ fusion (pure python)
def align(p, thr):
    """Map a probability so its own decision threshold lands on 0.5 (piecewise linear)."""
    thr = min(max(thr, 1e-3), 1 - 1e-3)
    return 0.5 * p / thr if p < thr else 0.5 + 0.5 * (p - thr) / (1 - thr)

def fuse(pv, pa, thv, tha, mode="avg"):
    """Fuse video/audio fake-probabilities. Either may be None (modality unavailable).
    Returns (fused_score, label); decision threshold is 0.5 on the aligned scale."""
    parts = []
    if pv is not None: parts.append(align(pv, thv))
    if pa is not None: parts.append(align(pa, tha))
    if not parts: return None, "UNKNOWN"
    score = max(parts) if mode == "max" else sum(parts) / len(parts)
    return score, ("FAKE" if score >= 0.5 else "REAL")


# ------------------------------------------------------------------ video
class VideoScorer:
    def __init__(self, model_path, device=None):
        import torch
        import deepfake_video_predict as dvp
        from facenet_pytorch import MTCNN
        self.torch, self.dvp = torch, dvp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = dvp.DeepfakeCNNLSTM(unfreeze_last_n_blocks=4)
        ck = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model = self.model.to(self.device).eval()
        self.detector = MTCNN(keep_all=True, device=self.device)

    def score(self, video_path, max_raw_frames=150):
        dvp, torch = self.dvp, self.torch
        try:
            frames = dvp.extract_frames(video_path, max_raw_frames=max_raw_frames)
            crops, flags = dvp.detect_and_crop_faces(frames, self.detector)
        except Exception as e:
            return dict(prob=None, error=str(e))
        if not crops:
            return dict(prob=None, error="no face detected", n_detected=0)
        seq = dvp.sample_sequence(crops, flags, dvp.SEQ_LEN)
        x = torch.stack([dvp.transform_eval(i) for i in seq]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()
        q = sum(flags) / len(flags)
        return dict(prob=p, error=None, n_detected=len(crops), quality=q,
                    reliable=q >= dvp.MIN_QUALITY_FRAME_FRACTION)


# ------------------------------------------------------------------ audio
def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

class AudioScorer:
    def __init__(self, ckpt_path, fix=False, floor=None, device=None, threshold=None):
        import torch, torchaudio
        import audio_branch as ab
        self.torch, self.ab, self.fix, self.floor = torch, ab, fix, floor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ab.build_model(pretrained=False)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        self.mel = torchaudio.transforms.MelSpectrogram(ab.SR, n_fft=ab.N_FFT, hop_length=ab.HOP, n_mels=ab.N_MELS)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.min_speech, self.min_dyn = 0.75, 12.0      # speech gate (seconds of speech, dB dynamics)
        cfg = Path(ckpt_path).parent / "audio_cfg.json"
        self.threshold = threshold if threshold is not None else \
            (json.load(open(cfg))["threshold"] if cfg.exists() else 0.5)

    def speech_check(self, y):
        """Is there clear speech? Stationary noise has little level variation; silence has no active frames."""
        n = len(y) // 160
        if n < 20: return False, 0.0, 0.0
        db = 20 * np.log10(np.sqrt((y[:n * 160].reshape(n, 160) ** 2).mean(1)) + 1e-6)
        sp = float((db > max(np.percentile(db, 5) + 12, -50)).sum() / 100)
        dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
        return (sp >= self.min_speech and dyn >= self.min_dyn), sp, dyn

    def _mel(self, video_path):
        ab, torch = self.ab, self.torch
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "a.wav")
            r = subprocess.run([_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video_path, "-vn",
                                "-af", "silenceremove=start_periods=1:start_threshold=-50dB",   # same as training
                                "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(ab.SR), "-t", "6", wav],
                               capture_output=True)
            if r.returncode != 0 or not os.path.exists(wav):
                return None, "no audio track", None
            with wave.open(wav, "rb") as w:
                y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if len(y) < 4000:
            return None, "audio too short / silent", None
        m = self.db(self.mel(torch.from_numpy(y)[None]))[0]
        m = ((m + 40) / 40).clamp(-1.0, 2.0)
        T = m.shape[1]
        m = m[:, :ab.STORE_FRAMES] if T >= ab.STORE_FRAMES else \
            torch.nn.functional.pad(m, (0, ab.STORE_FRAMES - T), value=-1.0)
        return m.numpy().astype(np.float16).astype(np.float32), None, y      # same fp16 rounding as stored mels

    def score(self, video_path, explain=False):
        ab, torch = self.ab, self.torch
        try:
            m, err, y = self._mel(video_path)
        except Exception as e:
            return dict(prob=None, error=str(e))
        if m is None:
            return dict(prob=None, error=err)
        ok, sp, dyn = self.speech_check(y)
        if not ok:
            return dict(prob=None, error=f"no clear speech found ({sp:.1f} s of speech, {dyn:.0f} dB dynamics)",
                        wave=y, raw_mel=m, crop=(0, 0))
        raw, crop = m, (0, 0)
        if self.fix:
            n = int(ab.STORE_FRAMES - (m == -1.0).all(axis=0).sum())
            if n < ab.FIX_CROP_FRAMES:
                return dict(prob=None, error=f"only {n/100:.1f}s of audio (need {ab.FIX_CROP_FRAMES/100:.0f}s)", wave=y, raw_mel=raw, crop=crop)
            off = (n - ab.FIX_CROP_FRAMES) // 2
            crop = (off, off + ab.FIX_CROP_FRAMES)
            m = m[:, off:off + ab.FIX_CROP_FRAMES]
            if self.floor:
                m = np.maximum(m, m.max() - self.floor / 40.0)
            m = m - m.mean(axis=1, keepdims=True)
        else:
            o = (ab.STORE_FRAMES - ab.CROP_FRAMES) // 2
            crop = (o, o + ab.CROP_FRAMES)
            m = m[:, o:o + ab.CROP_FRAMES]
        x = torch.from_numpy(np.ascontiguousarray(m))[None, None].to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()
        return dict(prob=p, error=None, wave=y, raw_mel=raw, crop=crop)


# ------------------------------------------------------------------ combined
class MultimodalDetector:
    def __init__(self, video_model, audio_model, audio_fix=False, floor=None,
                 video_threshold=0.30, audio_threshold=None, fusion="avg"):
        self.video = VideoScorer(video_model)
        self.audio = AudioScorer(audio_model, fix=audio_fix, floor=floor, threshold=audio_threshold)
        self.thv, self.tha, self.fusion = video_threshold, self.audio.threshold, fusion

    def analyze(self, path):
        v, a = self.video.score(path), self.audio.score(path)
        score, label = fuse(v["prob"], a["prob"], self.thv, self.tha, self.fusion)
        return dict(video=v, audio=a, fused_score=score, label=label, thv=self.thv, tha=self.tha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--video-model", required=True)
    ap.add_argument("--audio-model", required=True); ap.add_argument("--audio-fix", action="store_true")
    ap.add_argument("--floor", type=float, default=None); ap.add_argument("--fusion", default="avg")
    ap.add_argument("--video-threshold", type=float, default=0.30)
    a = ap.parse_args()
    d = MultimodalDetector(a.video_model, a.audio_model, a.audio_fix, a.floor, a.video_threshold, None, a.fusion)
    print(json.dumps(d.analyze(a.video), indent=2, default=str))