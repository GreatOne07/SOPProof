"""Generate the precomputed SOPProof guest-demo dataset.

Usage:
    .venv/bin/python demo_data/generate_demo_data.py

Everything produced here is DEMO DATA: precomputed demonstration records,
not live inspections. No real user uploads, reports or worker data are
used. Photo evidence A is the repository's own public sample photograph
(hardhat_sample.jpg, also served at /hero-photo.jpg). Photo evidence B is
a committed AI-generated demo photograph (demo_photo_b_source.jpg) that
illustrates a potential missing-helmet scenario. Video evidence frames
and the demo clip are derived from the repository's sample footage
(hardhat_video.mp4). Every derived image carries a baked-in DEMO DATA
label, reports are rendered with the real SOPProof report generators
from precomputed inputs, then stamped with a DEMO DATA banner. Guest
sessions may only ever read this directory.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo_data"
sys.path.insert(0, str(PROJECT_ROOT))

BANNER = (
    '<div style="background:#7c2d12;color:#fff;padding:14px 18px;'
    'border-radius:10px;margin-bottom:22px;font-weight:700;letter-spacing:.04em">'
    "DEMO DATA — Synthetic demonstration report. "
    "This is not a real inspection and contains no real worker data.</div>"
)


def stamp_demo_banner(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    if "DEMO DATA" not in html:
        html = html.replace('<div class="container">', '<div class="container">' + BANNER, 1)
        html_path.write_text(html, encoding="utf-8")


def stamp_demo_label(img: np.ndarray) -> np.ndarray:
    """Bake a small DEMO DATA pill into the bottom-left corner.

    Labelling only — never boxes, masks or scores. The pill sits in the
    corner so every surface that shows the image (cards, detail view,
    raw report HTML) keeps the demo identification.
    """
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.55, min(1.0, w / 1200.0))
    text = "DEMO DATA - PRECOMPUTED EXAMPLE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    pad_x, pad_y = int(14 * scale) + 4, int(10 * scale) + 4
    x0, y1 = int(16 * scale) + 2, h - (int(16 * scale) + 2)
    x1, y0 = x0 + tw + pad_x * 2, y1 - th - pad_y * 2
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 10, 12), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
    cv2.rectangle(out, (x0, y0), (x1, y1), (31, 111, 255), max(1, int(2 * scale)))
    cv2.putText(out, text, (x0 + pad_x, y1 - pad_y),
                font, scale, (235, 240, 248), 2, cv2.LINE_AA)
    return out


def save_demo_image(img: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), stamp_demo_label(img),
                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Could not write demo image {path}")


def cover_resize(img: np.ndarray, size=(1200, 800)) -> np.ndarray:
    """Resize + centre-crop to size (like CSS object-fit: cover)."""
    tw, th = size
    h, w = img.shape[:2]
    s = max(tw / w, th / h)
    resized = cv2.resize(img, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    x, y = (rw - tw) // 2, (rh - th) // 2
    return resized[y:y + th, x:x + tw]


def photo_evidence(source: Path, path: Path, size=None) -> None:
    """Copy a realistic photograph into the demo dataset, DEMO-labelled."""
    img = cv2.imread(str(source))
    if img is None:
        raise RuntimeError(f"Demo photo source could not be read: {source}")
    if size is not None:
        img = cover_resize(img, size)
    save_demo_image(img, path)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError(
        "ffmpeg is required to derive demo video frames "
        "(install imageio-ffmpeg or ffmpeg).")


def video_frame(source: Path, seconds: float, path: Path) -> None:
    """Extract one genuine frame from the sample footage, DEMO-labelled."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "frame.png"
        subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-ss", str(seconds),
             "-i", str(source), "-frames:v", "1", str(raw)],
            check=True, capture_output=True)
        img = cv2.imread(str(raw))
    if img is None:
        raise RuntimeError(f"Could not extract frame at {seconds}s from {source}")
    save_demo_image(cover_resize(img), path)


def demo_video(frame_paths, path: Path) -> None:
    """Build the playable 3.0s demo MP4 from the two review frames.

    Frame 0 is held until its 1.667s review timestamp, then frame 50,
    matching video_inspection.json. H.264 via ffmpeg so browsers play
    it natively; falls back to the historical MJPG + avconvert chain.
    """
    images = [cv2.imread(str(p)) for p in frame_paths]
    if any(img is None for img in images):
        raise RuntimeError("Demo frame image could not be read.")
    height, width = images[0].shape[:2]
    # 10 fps x 3.0s; frame 50's stored timestamp is 1.667s.
    sequence = [images[0]] * 17 + [images[-1]] * 13

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            seq_dir = Path(tmp)
            for i, frame in enumerate(sequence):
                cv2.imwrite(str(seq_dir / f"f{i:03d}.png"), frame)
            subprocess.run(
                [_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "10",
                 "-i", str(seq_dir / "f%03d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(path)],
                check=True, capture_output=True)
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            avi = tmp / "frames.avi"
            writer = cv2.VideoWriter(
                str(avi), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not create demo video.")
            for frame in sequence:
                writer.write(frame if frame.shape[:2] == (height, width)
                             else cv2.resize(frame, (width, height)))
            writer.release()
            if not avi.is_file() or avi.stat().st_size == 0:
                raise RuntimeError("Could not create demo video.")
            if shutil.which("avconvert"):
                mov = tmp / "encoded.mov"
                subprocess.run(
                    ["avconvert", "--source", str(avi), "--preset",
                     "PresetHighestQuality", "--output", str(mov), "--replace"],
                    check=True, capture_output=True)
                subprocess.run(
                    ["avconvert", "--source", str(mov), "--preset",
                     "PresetPassthrough", "--output", str(path), "--replace"],
                    check=True, capture_output=True)
            else:
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"avc1"), 10, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Could not create demo video.")
                for frame in sequence:
                    writer.write(frame if frame.shape[:2] == (height, width)
                                 else cv2.resize(frame, (width, height)))
                writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Could not create demo video.")


def main() -> None:
    from backend.report import generate_html_report
    import backend.video_report as video_report

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    # ---- realistic evidence images (12-hex names so existing parsing works)
    # A: repository sample photograph — three workers, helmets visible.
    photo_evidence(PROJECT_ROOT / "hardhat_sample.jpg",
                   DEMO_DIR / "de0000000009.jpg")
    # B: committed AI-generated demo photograph — one worker bare-headed,
    # illustrating the potential missing-helmet scenario in report 2.
    photo_evidence(DEMO_DIR / "demo_photo_b_source.jpg",
                   DEMO_DIR / "de0000000010.jpg", size=(1200, 800))

    # ---- demo photo reports (real generator, precomputed inputs) ----
    base_people = [
        {"person_number": n, "head_visibility": "clear", "helmet_observation": "visible_on_head",
         "assessment": "helmet_visible",
         "evidence": "DEMO: Precomputed example — a helmet is visible on this worker in the demo photograph."}
        for n in (1, 2, 3)
    ]
    generate_html_report({
        "source_image": "de0000000009.jpg",
        "image_summary": "DEMO DATA: Precomputed demonstration using the repository sample photograph. No live inspection took place.",
        "people": base_people,
        "overall_result": "all_helmets_visible",
        "human_review_required": False,
        "detection_summary": {"people_count": 3, "helmet_count": 3, "matched_count": 3, "review_count": 0},
        "limitations": ["DEMO DATA — precomputed demonstration report, not a real inspection."],
    }, DEMO_DIR / "de0000000001_report.html")

    generate_html_report({
        "source_image": "de0000000010.jpg",
        "image_summary": "DEMO DATA: Precomputed demonstration using an AI-generated demo photograph, showing how a potential concern is presented for human review.",
        "people": [
            {"person_number": 1, "head_visibility": "clear", "helmet_observation": "not_visible",
             "assessment": "potential_missing_helmet",
             "evidence": "DEMO: Precomputed example — the worker on the right has no helmet, so this is flagged for human verification."},
            {"person_number": 2, "head_visibility": "partially_obscured", "helmet_observation": "uncertain",
             "assessment": "uncertain",
             "evidence": "DEMO: Precomputed example of an uncertain observation."},
        ],
        "overall_result": "potential_missing_helmet",
        "human_review_required": True,
        "detection_summary": {"people_count": 2, "helmet_count": 0, "matched_count": 0, "review_count": 2},
        "limitations": ["DEMO DATA — precomputed demonstration report, not a real inspection."],
    }, DEMO_DIR / "de0000000002_report.html")

    # ---- demo video report (real generator, precomputed inputs) ----
    job = DEMO_DIR / "de0000000003"
    vout = job / "outputs"
    frames_dir = vout / "video_review_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Genuine frames from the repository sample footage at the stored
    # review timestamps (0.0s and 1.667s of the demo timeline).
    video_frame(PROJECT_ROOT / "hardhat_video.mp4", 0.0, frames_dir / "frame_0000.jpg")
    video_frame(PROJECT_ROOT / "hardhat_video.mp4", 1.667, frames_dir / "frame_0050.jpg")
    demo_frames = [frames_dir / "frame_0000.jpg", frames_dir / "frame_0050.jpg"]
    demo_video(demo_frames, job / "demo-video.mp4")
    demo_video(demo_frames, vout / "sopproof_video_preview.mp4")

    frame_record = lambda n, ts, overall, summary: {
        "frame": n, "timestamp_seconds": ts,
        "assessment": {"overall_result": overall, "image_summary": summary, "people": [
            {"assessment": "helmet_visible" if overall == "all_helmets_visible" else "uncertain",
             "evidence": "DEMO: Precomputed frame assessment for the guest demonstration."}]},
    }
    (vout / "video_inspection.json").write_text(json.dumps({
        "source_video": "demo-video.mp4",
        "overall_assessment": "DEMO DATA: Review completed for demonstration.",
        "human_review_required": True,
        "duration_seconds": 3.0, "total_frames": 90,
        "successful_frame_reviews": 2, "failed_frame_reviews": 0,
        "frames": [
            frame_record(0, 0.0, "all_helmets_visible", "DEMO: Precomputed opening frame from sample footage."),
            frame_record(50, 1.667, "uncertain", "DEMO: Precomputed frame routed to human review."),
        ],
        "limitations": ["DEMO DATA — precomputed demonstration report, not a real inspection."],
    }, indent=2), encoding="utf-8")

    video_report.PROJECT_ROOT = job
    video_report.OUTPUT_DIR = vout
    video_report.INPUT_JSON = vout / "video_inspection.json"
    video_report.OUTPUT_HTML = vout / "video_inspection_report.html"
    video_report.ANNOTATED_VIDEO = vout / "sopproof_video_preview.mp4"
    video_report.generate_report()

    for html in (DEMO_DIR / "de0000000001_report.html",
                 DEMO_DIR / "de0000000002_report.html",
                 vout / "video_inspection_report.html"):
        stamp_demo_banner(html)

    print("Demo dataset generated in", DEMO_DIR)


if __name__ == "__main__":
    main()
