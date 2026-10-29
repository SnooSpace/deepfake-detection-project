"""
Audio deepfake branch (FakeAVCeleb) - mirrors the video CNN-LSTM.

Pipeline: mp4 -> 16 kHz mono wav -> log-mel (128 x T) -> 24 overlapping chunks
          -> EfficientNet-B0 (last 4 blocks unfrozen) -> LSTM(256) -> logit (FAKE=1)

Usage (Colab):
    !python audio_branch.py extract --root /content/FakeAVCeleb --out /content/mels
    !python audio_branch.py train   --mels /content/mels --out /content/audio_run
"""
import argparse, os, re, subprocess, json
from pathlib import Path
import numpy as np
import pandas as pd

SR, N_MELS, HOP, N_FFT = 16000, 128, 160, 400      # 10 ms hop -> 100 frames/s
STORE_FRAMES = 600                                  # store 6 s
CROP_FRAMES = 400                                   # model sees 4 s
SEQ_LEN, CHUNK_FRAMES = 24, 64                      # 24 chunks x 0.64 s, overlapping
CATEGORIES = ["RealVideo-RealAudio", "RealVideo-FakeAudio",
              "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]


# ----------------------------------------------------------------- extraction
def build_index(root):
    """Scan FakeAVCeleb folder -> DataFrame(path, category, speaker, audio_label, video_label)."""
    rows = []
    for p in Path(root).rglob("*.mp4"):
        parts = p.parts
        if "frames" in parts or "moved" in parts:   # extra folders in some Kaggle mirrors
            continue
        cat = next((c for c in CATEGORIES if c in parts), None)
        spk = next((x for x in parts if re.fullmatch(r"id\d+", x)), None)
        if cat is None or spk is None:
            continue
        rows.append(dict(path=str(p), category=cat, speaker=spk,
                         audio_label=int("FakeAudio" in cat),
                         video_label=int(cat.startswith("FakeVideo"))))
    df = pd.DataFrame(rows)
    df["clip_id"] = [f"{r.category}__{r.speaker}__{Path(r.path).stem}" for r in df.itertuples()]
    return df.drop_duplicates("clip_id").reset_index(drop=True)


def extract_mels(root, out, limit=None):
    import torch, torchaudio
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    out = Path(out); (out / "mels").mkdir(parents=True, exist_ok=True)
    df = build_index(root)
    if limit: df = df.sample(limit, random_state=0).reset_index(drop=True)
    print(df.category.value_counts(), "\nspeakers:", df.speaker.nunique())
    keep = []
    for r in df.itertuples():
        npy = out / "mels" / f"{r.clip_id}.npy"
        if not npy.exists():
            wav = Path("/tmp/tmp.wav")   # local disk, even if --out is on Drive
            # silenceremove: FakeAVCeleb fake-audio clips have a known leading-silence
            # shortcut (arXiv 2506.05851); trimming it forces the model to learn real artifacts.
            ok = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", r.path, "-vn",
                                 "-af", "silenceremove=start_periods=1:start_threshold=-50dB",
                                 "-ac", "1", "-ar", str(SR), "-t", "6", str(wav)]).returncode == 0
            if not ok or not wav.exists():
                print("skip (no audio?):", r.path); continue
            y, _ = torchaudio.load(str(wav))
            m = db(mel(y))[0]                                   # (128, T) in dB, floor at max-80
            m = ((m + 40) / 40).clamp(-1.0, 2.0)                # rough scaling to ~[-1, 2]
            T = m.shape[1]
            m = m[:, :STORE_FRAMES] if T >= STORE_FRAMES else \
                torch.nn.functional.pad(m, (0, STORE_FRAMES - T), value=-1.0)
            np.save(npy, m.numpy().astype(np.float16))
        keep.append(r.Index)
    df.loc[keep].to_csv(out / "index.csv", index=False)


# ------------------------------------------------------------------- splitting
def make_splits(df, seed=42):
    """Clip- AND speaker-level split. Splitting by speaker matters: FakeAVCeleb reuses
    the same source audio/identity across many clips, so a clip-only split leaks."""
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy(); df["split"] = "train"
    folds = list(StratifiedGroupKFold(6, shuffle=True, random_state=seed)
                 .split(df, df.audio_label, groups=df.speaker))
    df.loc[df.index[folds[0][1]], "split"] = "test"
    df.loc[df.index[folds[1][1]], "split"] = "val"
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not set(df[df.split == a].speaker) & set(df[df.split == b].speaker)
    return df


# --------------------------------------------------------------------- dataset
def _import_torch():
    import torch
    return torch

FIX_CROP_FRAMES = 300        # --fix: 3 s crop taken strictly inside the real (unpadded) audio

class MelDataset:
    def __init__(self, df, mel_dir, train, fix=False, floor=None):
        self.df, self.dir, self.train, self.fix = df.reset_index(drop=True), Path(mel_dir), train, fix
        self.floor = floor
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        torch = _import_torch()
        r = self.df.iloc[i]
        m = np.load(self.dir / "mels" / f"{r.clip_id}.npy").astype(np.float32)
        if self.fix:
            # (a) crop only from real audio -> clip length / padding can't leak the label
            n = int(r.n_valid)
            off = np.random.randint(0, n - FIX_CROP_FRAMES + 1) if self.train else (n - FIX_CROP_FRAMES) // 2
            m = m[:, off:off + FIX_CROP_FRAMES]
            if self.floor:   # raise everything quieter than (peak - X dB) to one level: hides
                             # noise-floor / digital-silence differences between real and fake
                x = self.floor + (np.random.uniform(-5, 5) if self.train else 0.0)
                m = np.maximum(m, m.max() - x / 40.0)
            # (b) per-band mean removal (cepstral-mean-style): kills the static spectral
            #     envelope / loudness / channel fingerprint that gave AUC 0.994 on band means
            m = m - m.mean(axis=1, keepdims=True)
            fill = 0.0
        else:
            off = np.random.randint(0, STORE_FRAMES - CROP_FRAMES + 1) if self.train \
                  else (STORE_FRAMES - CROP_FRAMES) // 2
            m = m[:, off:off + CROP_FRAMES]
            fill = -1.0
        if self.train:                                          # SpecAugment (light)
            T = m.shape[1]
            f = np.random.randint(0, 16); f0 = np.random.randint(0, N_MELS - f)
            t = np.random.randint(0, 40); t0 = np.random.randint(0, T - t)
            m[f0:f0 + f, :] = fill; m[:, t0:t0 + t] = fill
        return torch.from_numpy(m)[None], torch.tensor(float(r.audio_label))


# ----------------------------------------------------------------------- model
def build_model(unfreeze_last_n_blocks=4, hidden=256, pretrained=True):
    import torch, torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    class AudioCNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None).features
            for p in self.cnn.parameters(): p.requires_grad = False
            for blk in list(self.cnn)[-unfreeze_last_n_blocks:]:
                for p in blk.parameters(): p.requires_grad = True
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.lstm = nn.LSTM(1280, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)   # TODO: copy exact head/dropout from the video model

        def forward(self, x):                                   # x: (B,1,128,400)
            B, T = x.shape[0], x.shape[-1]
            starts = torch.linspace(0, T - CHUNK_FRAMES, SEQ_LEN).long().tolist()
            c = torch.stack([x[..., s:s + CHUNK_FRAMES] for s in starts], 1)   # (B,24,1,128,64)
            c = c.flatten(0, 1).repeat(1, 3, 1, 1)                             # (B*24,3,128,64)
            f = self.pool(self.cnn(c)).flatten(1).view(B, SEQ_LEN, -1)
            _, (h, _) = self.lstm(f)
            return self.head(h[-1]).squeeze(-1)                 # logit, FAKE = 1
    return AudioCNNLSTM()


# ------------------------------------------------------------------ train/eval
def predict(model, loader, device):
    import torch
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(torch.sigmoid(model(x.to(device))).cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()

def sweep(probs, y):
    from sklearn.metrics import balanced_accuracy_score
    ths = np.arange(0.05, 0.96, 0.05)
    scores = [balanced_accuracy_score(y, probs >= t) for t in ths]
    return float(ths[int(np.argmax(scores))]), ths, scores

def report(probs, y, th, name):
    from sklearn.metrics import classification_report, roc_auc_score
    print(f"\n== {name} @ threshold {th:.2f}  AUC={roc_auc_score(y, probs):.4f}")
    print(classification_report(y, probs >= th, target_names=["REAL", "FAKE"], digits=3))

def train(mels, out, epochs=15, lr=3e-5, bs=4, patience=4, limit=None, fix=False, floor=None):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(mels) / "index.csv")
    ok = [(Path(mels) / "mels" / f"{c}.npy").exists() for c in df.clip_id]
    if not all(ok): print(f"skipping {len(ok) - sum(ok)} clips with no .npy file")
    df = df[ok].reset_index(drop=True)
    if limit: df = df.sample(limit, random_state=0).reset_index(drop=True)
    if fix:
        nv = [int(STORE_FRAMES - (np.load(Path(mels) / "mels" / f"{c}.npy") == -1.0).all(axis=0).sum())
              for c in df.clip_id]
        df["n_valid"] = nv
        short = df[df.n_valid < FIX_CROP_FRAMES]
        print(f"--fix: dropping {len(short)} clips with < {FIX_CROP_FRAMES/100:.0f}s of audio "
              f"(by label: {short.audio_label.value_counts().to_dict()})")
        df = df[df.n_valid >= FIX_CROP_FRAMES].reset_index(drop=True)
    df = make_splits(df)
    print(df.groupby(["split", "audio_label"]).size().unstack())
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    n_real, n_fake = (tr.audio_label == 0).sum(), (tr.audio_label == 1).sum()
    pw = float((n_real / n_fake) ** 0.5)                        # softened, same rule as video
    print(f"pos_weight = sqrt({n_real}/{n_fake}) = {pw:.3f}")
    mk = lambda d, t: DataLoader(MelDataset(d, mels, t, fix, floor), batch_size=bs, shuffle=t, num_workers=2)
    L_tr, L_va, L_te = mk(tr, True), mk(va, False), mk(te, False)

    model = build_model().to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    best, bad = -1, 0
    for ep in range(epochs):
        model.train(); tot = 0
        for x, y in L_tr:
            opt.zero_grad(); loss = crit(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        pv, yv = predict(model, L_va, dev); auc = roc_auc_score(yv, pv)
        print(f"ep {ep+1:02d} train_loss={tot/len(tr):.4f} val_auc={auc:.4f}")
        if auc > best:
            best, bad = auc, 0; torch.save(model.state_dict(), out / "audio_best.pt")
        else:
            bad += 1
            if bad >= patience: break

    model.load_state_dict(torch.load(out / "audio_best.pt"))
    pv, yv = predict(model, L_va, dev)
    th, _, _ = sweep(pv, yv)                                    # threshold from VAL only
    pt, yt = predict(model, L_te, dev)
    report(pt, yt, th, "AUDIO test")
    json.dump({"threshold": th, "pos_weight": pw}, open(out / "audio_cfg.json", "w"))
    for name, d, p in (("train", tr, None), ("val", va, pv), ("test", te, pt)):
        if p is None: p, _ = predict(model, mk(d, False), dev)
        o = d[["clip_id", "category", "speaker", "audio_label", "video_label", "split"]].copy()
        o["audio_prob"] = p; o.to_csv(out / f"audio_scores_{name}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--root", required=True); e.add_argument("--out", required=True); e.add_argument("--limit", type=int, default=None)
    t = sp.add_parser("train"); t.add_argument("--mels", required=True); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=15); t.add_argument("--limit", type=int, default=None)
    t.add_argument("--fix", action="store_true", help="remove length + spectral-envelope shortcuts")
    t.add_argument("--floor", type=float, default=None, help="with --fix: clamp anything X dB below clip peak")
    a = ap.parse_args()
    extract_mels(a.root, a.out, a.limit) if a.cmd == "extract" else train(a.mels, a.out, epochs=a.epochs, limit=a.limit, fix=a.fix, floor=a.floor)
