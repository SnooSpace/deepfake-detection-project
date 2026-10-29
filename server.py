"""
Forensic-lab web UI. Run from the project folder:
  python server.py --audio-model audio_model/audio_best.pt --audio-fix --floor 30 --fusion max --video-threshold 0.15
Needs: multimodal.py, audio_branch.py, deepfake_video_predict.py, FINAL_model.pt, index.html (+ optional demo_clips/)
"""
import argparse, base64, os, tempfile, threading, uuid, webbrowser
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from multimodal import MultimodalDetector, fuse

ap = argparse.ArgumentParser()
ap.add_argument("--video-model", default="FINAL_model.pt"); ap.add_argument("--audio-model", required=True)
ap.add_argument("--audio-fix", action="store_true"); ap.add_argument("--floor", type=float, default=None)
ap.add_argument("--video-threshold", type=float, default=0.30)
ap.add_argument("--voice-advisory", action="store_true", help="never let the voice branch alone decide, even for sample clips")
ap.add_argument("--min-speech-sec", type=float, default=0.75); ap.add_argument("--min-dyn-db", type=float, default=12.0)
ap.add_argument("--audio-threshold", type=float, default=None)
ap.add_argument("--fusion", default="max", choices=["avg", "max"]); ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

HERE, DEMOS = Path(__file__).parent, Path(__file__).parent / "demo_clips"
print("Loading models...")
det = MultimodalDetector(args.video_model, args.audio_model, args.audio_fix, args.floor,
                         args.video_threshold, args.audio_threshold, args.fusion)
dvp = det.video.dvp
det.audio.min_speech, det.audio.min_dyn = args.min_speech_sec, args.min_dyn_db
app, JOBS, LOCK = FastAPI(), {}, threading.Lock()


def jpg(rgb, w=None, q=72):
    rgb = np.ascontiguousarray(rgb)
    if w and rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, int(rgb.shape[0] * w / rgb.shape[1])))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def gradcam(model, x):
    """Grad-CAM on the last conv block of EfficientNet, one map per sampled frame,
    explaining the FAKE logit (warm = pushed the score toward FAKE)."""
    acts, grads = {}, {}
    def fh(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))
    h = model.cnn[-1].register_forward_hook(fh)
    try:
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            logit = model(x)
            model.zero_grad(); logit.sum().backward()
    finally:
        h.remove()
    A, G = acts["a"].detach(), grads["g"]
    cam = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1))                    # (24,7,7)
    imp = cam.mean((1, 2)); imp = (imp / (imp.max() + 1e-8)).cpu().numpy()
    cam = F.interpolate(cam[:, None], size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    lo, hi = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    return torch.sigmoid(logit).item(), ((cam - lo) / (hi - lo + 1e-8)).cpu().numpy(), imp


def overlay(face, c, w):
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return jpg((face * 0.55 + heat * 0.45).astype(np.uint8), w)


def run_job(job, path, trusted):
    emit = lambda stage, **kw: job["events"].append(dict(stage=stage, **kw))
    V, A, pv = det.video, det.audio, None
    with LOCK:
        try:
            emit("start")
            try:
                frames = dvp.extract_frames(path, max_raw_frames=150)
                idx = np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)
                emit("frames", total=len(frames), thumbs=[jpg(frames[i], 200) for i in idx])
                crops, flags = dvp.detect_and_crop_faces(frames, V.detector)
                pick = np.linspace(0, max(len(crops) - 1, 0), min(12, len(crops))).astype(int) if crops else []
                emit("faces", detected=len(crops), good=int(sum(flags)),
                     crops=[jpg(np.array(crops[i].convert("RGB")), 120) for i in pick])
                if not crops:
                    emit("video", prob=None, thr=det.thv, error="No face detected")
                else:
                    seq = dvp.sample_sequence(crops, flags, dvp.SEQ_LEN)
                    x = torch.stack([dvp.transform_eval(i) for i in seq]).unsqueeze(0).to(V.device)
                    pv, cam, imp = gradcam(V.model, x)
                    emit("video", prob=pv, thr=det.thv, reliable=sum(flags) / len(flags) >= dvp.MIN_QUALITY_FRAME_FRACTION)
                    orig = [np.array(i.convert("RGB").resize((224, 224))) for i in seq]
                    b = int(imp.argmax())
                    emit("gradcam", orig=[jpg(o, 150) for o in orig], heat=[overlay(o, c, 150) for o, c in zip(orig, cam)],
                         imp=imp.tolist(), best=b, hero=overlay(orig[b], cam[b], 380), hero_orig=jpg(orig[b], 380))
            except Exception as e:
                emit("video", prob=None, thr=det.thv, error=str(e))
            a = A.score(path, explain=True)
            ev = dict(prob=a["prob"], thr=det.tha, error=a.get("error"))
            if a.get("wave") is not None and len(a["wave"]):
                env = [float(c.max()) if len(c) else 0.0 for c in np.array_split(np.abs(a["wave"]), 300)]
                top = max(env) or 1.0
                u8 = np.clip((a["raw_mel"] + 1) / 3 * 255, 0, 255).astype(np.uint8)
                ev.update(wave=[round(e / top, 3) for e in env], mel=base64.b64encode(u8.tobytes()).decode(),
                          mel_shape=list(u8.shape), crop=list(a["crop"]), secs=len(a["wave"]) / 16000)
            emit("audio", **ev)
            pa = a["prob"]
            fv, fa = pv is not None and pv >= det.thv, pa is not None and pa >= det.tha
            score, label = fuse(pv, pa, det.thv, det.tha, det.fusion)
            if pv is None and pa is None:
                label, basis, why = "UNKNOWN", "none", "No face and no clear speech were found, so nothing could be analysed."
            elif pa is None:
                basis = "video"
                why = ("The face looks manipulated. " if fv else "The face looks genuine. ") + \
                      f"The voice was not used ({a.get('error') or 'no audio'}), so this rests on the video alone."
            elif pv is None:
                basis = "voice"
                why = "No face could be analysed, so this rests on the voice alone."
                if not trusted: label = "UNSURE"
            else:
                basis = "both"
                if label == "REAL": why = "Neither the face nor the voice showed signs of manipulation."
                elif fv and fa: why = "Both the face and the voice look manipulated."
                elif fv: why = "The face looks manipulated, while the voice sounds genuine."
                else: why = "The voice looks synthetic, while the face looks genuine."
            detail = (f"Video {pv:.2f} (threshold {det.thv:.2f})" if pv is not None else "Video not used") + " · " + \
                     (f"Voice {pa:.2f} (threshold {det.tha:.2f})" if pa is not None else "Voice not used")
            emit("fusion", score=score, label=label, why=why, detail=detail, basis=basis, mode=det.fusion)
        except Exception as e:
            emit("error", message=str(e))
        finally:
            job["done"] = True


def start(path, cleanup, trusted=False):
    jid, job = uuid.uuid4().hex[:8], dict(events=[], done=False)
    JOBS[jid] = job
    def work():
        run_job(job, path, trusted)
        if cleanup: os.unlink(path)
    threading.Thread(target=work, daemon=True).start()
    return dict(id=jid)


@app.get("/")
def index(): return FileResponse(HERE / "index.html")

@app.get("/logo.png")
def logo():
    p = HERE / "logo.png"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)

@app.get("/logo.svg")
def logo_svg():
    p = HERE / "logo.svg"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="image/svg+xml")

@app.get("/three.min.js")
def three_js():
    p = HERE / "three.min.js"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="application/javascript")

@app.get("/head_model.js")
def head_model():
    p = HERE / "head_model.js"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="application/javascript")

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "v.mp4").suffix or ".mp4")
    tmp.write(await file.read()); tmp.close()
    return start(tmp.name, True, False)       # uploads: voice branch is advisory

@app.get("/api/events/{jid}")
def events(jid: str, after: int = 0):
    job = JOBS.get(jid)
    if job is None: raise HTTPException(404)
    ev = job["events"]
    return dict(events=ev[after:], n=len(ev), done=job["done"])

@app.get("/api/demos")
def demos(): return sorted(p.name for p in DEMOS.glob("*.mp4")) if DEMOS.exists() else []

@app.post("/api/demo/{name}")
def demo(name: str):
    p = DEMOS / Path(name).name
    if not p.exists(): raise HTTPException(404)
    return start(str(p), False, not args.voice_advisory)

@app.get("/demo/{name}")
def demo_file(name: str):
    p = DEMOS / Path(name).name
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p)


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")