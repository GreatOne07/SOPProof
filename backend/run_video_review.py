import json
import os
from pathlib import Path

job_value = os.environ.get("SOPPROOF_VIDEO_JOB_DIR")

if not job_value:
    raise RuntimeError("SOPPROOF_VIDEO_JOB_DIR is missing.")

job_root = Path(job_value).resolve()
outputs = job_root / "outputs"
video_path = job_root / "hardhat_video.mp4"

if not video_path.exists():
    raise FileNotFoundError(video_path)

outputs.mkdir(parents=True, exist_ok=True)

# Run the existing Muse Spark video inspection engine.
import backend.video_inspection as video_inspection

video_inspection.PROJECT_ROOT = job_root
video_inspection.VIDEO_PATH = video_path
video_inspection.PERSON_CSV = outputs / "video_person_detections.csv"
video_inspection.HELMET_CSV = outputs / "video_helmet_detections.csv"
video_inspection.ASSOCIATION_CSV = (
    outputs / "video_worker_helmet_association.csv"
)
video_inspection.OUTPUT_DIR = outputs
video_inspection.FRAME_DIR = outputs / "video_review_frames"
video_inspection.OUTPUT_JSON = outputs / "video_inspection.json"
video_inspection.OUTPUT_TEXT = (
    outputs / "video_inspection_timeline.txt"
)

video_inspection.inspect_video()

# Store the original uploaded filename in the report data.
original_video_name = os.environ.get("SOPPROOF_ORIGINAL_VIDEO_NAME")
metadata_path = outputs / "video_inspection.json"

if original_video_name and metadata_path.is_file():
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    metadata["source_video"] = Path(original_video_name).name
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


# Generate this job's own annotated preview, inside its own outputs
# directory, from this job's detections. Best-effort: a preview failure
# must not discard an otherwise successful inspection, and the report
# already handles a missing preview explicitly.
def generate_annotated_preview():
    import csv
    from collections import defaultdict

    import cv2

    def load(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def by_frame(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row["frame"])].append(row)
        return grouped

    people = by_frame(load(outputs / "video_person_detections.csv"))
    helmets = by_frame(load(outputs / "video_helmet_detections.csv"))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open source video for annotated preview.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prefer H.264 (avc1) so the preview plays in browsers; fall back to
    # other codecs this OpenCV build supports. The report references
    # whichever per-job file is actually produced.
    writer = None
    preview_path = None

    for fourcc, suffix in (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")):
        candidate = outputs / f"sopproof_video_preview{suffix}"
        attempt = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
        )

        if attempt.isOpened():
            writer = attempt
            preview_path = candidate
            break

        attempt.release()

    if writer is None or preview_path is None:
        cap.release()
        raise RuntimeError("Cannot create annotated preview video.")

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            for person in people.get(frame_number, []):
                x1, y1 = int(float(person["left"])), int(float(person["top"]))
                x2, y2 = int(float(person["right"])), int(float(person["bottom"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 3)
                cv2.putText(
                    frame,
                    str(person["track_id"]),
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 120, 0),
                    2,
                )

            for helmet in helmets.get(frame_number, []):
                x1, y1 = int(float(helmet["left"])), int(float(helmet["top"]))
                x2, y2 = int(float(helmet["right"])), int(float(helmet["bottom"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 3)
                cv2.putText(
                    frame,
                    str(helmet["track_id"]),
                    (x1, min(height - 10, y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 220, 60),
                    2,
                )

            cv2.putText(
                frame,
                "SOPPROOF | ANNOTATED PREVIEW - VISUAL REVIEW REQUIRED",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            frame_number += 1
    finally:
        cap.release()
        writer.release()

    if frame_number == 0 or not preview_path.is_file():
        raise RuntimeError("Annotated preview produced no frames.")

    print("Annotated preview:", preview_path)
    return preview_path


annotated_preview_path = outputs / "sopproof_video_preview.mp4"

try:
    annotated_preview_path = generate_annotated_preview()
except Exception as preview_error:
    print("ANNOTATED PREVIEW UNAVAILABLE:", repr(preview_error))


# Generate the HTML report using the existing report generator.
import backend.video_report as video_report

video_report.PROJECT_ROOT = job_root
video_report.OUTPUT_DIR = outputs
video_report.INPUT_JSON = outputs / "video_inspection.json"
video_report.OUTPUT_HTML = outputs / "video_inspection_report.html"
video_report.ANNOTATED_VIDEO = annotated_preview_path

video_report.generate_report()

report_path = outputs / "video_inspection_report.html"

if not report_path.exists():
    raise RuntimeError("Video HTML report was not generated.")

print("\nVIDEO REVIEW AND REPORT COMPLETED")
print("Report:", report_path)
