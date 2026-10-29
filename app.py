"""
Gradio demo: upload a video -> video score, audio score, fused verdict.

pip install gradio imageio-ffmpeg torchaudio   (plus the packages you already use)
python app.py --video-model FINAL_model.pt --audio-model audio_run_fixed/audio_best.pt --audio-fix --floor 30
"""
import argparse, html
import gradio as gr
from multimodal import MultimodalDetector

ap = argparse.ArgumentParser()
ap.add_argument("--video-model", default="FINAL_model.pt")
ap.add_argument("--audio-model", required=True)
ap.add_argument("--audio-fix", action="store_true", help="use if the audio model was trained with --fix")
ap.add_argument("--floor", type=float, default=None)
ap.add_argument("--video-threshold", type=float, default=0.30)
ap.add_argument("--audio-threshold", type=float, default=None, help="default: read audio_cfg.json")
ap.add_argument("--fusion", default="max", choices=["avg", "max"])
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

print("Loading models (first run may take a minute)...")
det = MultimodalDetector(args.video_model, args.audio_model, args.audio_fix, args.floor,
                         args.video_threshold, args.audio_threshold, args.fusion)

def bar(name, p, thr, note=""):
    if p is None:
        return f"<div style='margin:10px 0'><b>{name}</b>: <i>unavailable</i> — {html.escape(note)}</div>"
    fake = p >= thr
    col = "#d9534f" if fake else "#2e9e5b"
    return (f"<div style='margin:10px 0'><b>{name}</b>: {'FAKE' if fake else 'REAL'} "
            f"<span style='opacity:.7'>(fake prob {p:.2f}, threshold {thr:.2f})</span>"
            f"<div style='background:#8884;border-radius:6px;height:14px;position:relative;margin-top:4px'>"
            f"<div style='width:{p*100:.0f}%;background:{col};height:14px;border-radius:6px'></div>"
            f"<div style='position:absolute;left:{thr*100:.0f}%;top:-3px;width:2px;height:20px;background:#888'></div>"
            f"</div></div>")

def run(video):
    if video is None:
        return "<i>Upload a video first.</i>"
    r = det.analyze(video)
    v, a = r["video"], r["audio"]
    label, score = r["label"], r["fused_score"]
    col = {"FAKE": "#d9534f", "REAL": "#2e9e5b"}.get(label, "#888")
    out = f"<h2 style='color:{col};margin-bottom:0'>Verdict: {label}</h2>"
    if score is not None:
        out += f"<div style='opacity:.7'>fusion = {args.fusion}, aligned score {score:.2f} (FAKE if ≥ 0.50)</div>"
    out += bar("Video branch (CNN-LSTM)", v["prob"], r["thv"], v.get("error") or "")
    out += bar("Audio branch (mel CNN-LSTM)", a["prob"], r["tha"], a.get("error") or "")
    notes = []
    if v.get("prob") is not None and not v.get("reliable", True):
        notes.append("Face was partly blocked or unclear in many frames — video score may be unreliable.")
    out += "".join(f"<div style='font-size:.9em;opacity:.8;margin-top:8px'>⚠ {html.escape(n)}</div>" for n in notes)
    return out

with gr.Blocks(title="Deepfake detector: video + audio") as demo:
    gr.Markdown("# Deepfake detector — video + audio\nUpload a short clip with a visible face and speech.")
    inp = gr.Video(label="Video")
    btn = gr.Button("Analyze", variant="primary")
    res = gr.HTML()
    btn.click(run, inp, res)

demo.launch(share=args.share)
