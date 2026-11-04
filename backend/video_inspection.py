
import base64
import csv
import json
import cv2

from pathlib import Path

from backend.meta_client import describe_model_error, get_meta_client
from backend.inspection import MODEL, INSPECTION_SCHEMA


# =========================================
# 1. PROJECT CONFIGURATION
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "hardhat_video.mp4"

PERSON_CSV = PROJECT_ROOT / "outputs/video_person_detections.csv"
HELMET_CSV = PROJECT_ROOT / "outputs/video_helmet_detections.csv"
ASSOCIATION_CSV = PROJECT_ROOT / "outputs/video_worker_helmet_association.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "video_review_frames"

OUTPUT_JSON = OUTPUT_DIR / "video_inspection.json"
OUTPUT_TEXT = OUTPUT_DIR / "video_inspection_timeline.txt"

# Exactly five frames are reviewed, evenly distributed across the full
# video duration (first frame through final frame). See select_sample_frames.
SAMPLE_FRAME_COUNT = 5


def select_sample_frames(total_frames, frame_count=SAMPLE_FRAME_COUNT):
    """Return evenly spaced frame indices spanning the whole video.

    First index is 0 and the last is total_frames - 1, so sampled
    timestamps always reach the end of the video regardless of duration.
    """

    if total_frames <= 0:
        return []

    if total_frames <= frame_count:
        return list(range(total_frames))

    frames = []

    for index in range(frame_count):
        frame_number = round(
            index * (total_frames - 1) / (frame_count - 1)
        )

        if frame_number not in frames:
            frames.append(frame_number)

    return frames


# =========================================
# 2. LOAD CSV DETECTIONS
# =========================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required detection file missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


def group_by_frame(rows):

    grouped = {}

    for row in rows:

        frame_number = int(row["frame"])

        grouped.setdefault(frame_number, []).append(row)

    return grouped


# =========================================
# 3. PREPARE DETECTION CONTEXT
# =========================================

def prepare_frame_context(
    frame_number,
    people_by_frame,
    helmets_by_frame,
    associations_by_frame
):

    people = people_by_frame.get(frame_number, [])
    helmets = helmets_by_frame.get(frame_number, [])
    associations = associations_by_frame.get(frame_number, [])

    person_records = []

    for person in people:

        person_records.append({
            "provisional_track_id": person["track_id"],
            "bounding_box": {
                "left": int(person["left"]),
                "top": int(person["top"]),
                "right": int(person["right"]),
                "bottom": int(person["bottom"])
            },
            "confidence": float(person["confidence"])
        })

    helmet_records = []

    for helmet in helmets:

        helmet_records.append({
            "provisional_track_id": helmet["track_id"],
            "bounding_box": {
                "left": int(helmet["left"]),
                "top": int(helmet["top"]),
                "right": int(helmet["right"]),
                "bottom": int(helmet["bottom"])
            },
            "confidence": float(helmet["confidence"])
        })

    association_records = []

    for association in associations:

        association_records.append({
            "person_track_id": association["person_track_id"],
            "helmet_track_id": association["helmet_track_id"],
            "association_status": association["association_status"]
        })

    return {
        "frame": frame_number,
        "person_detections": person_records,
        "helmet_detections": helmet_records,
        "association_results": association_records,
        "tracking_limitation": (
            "Track identities are provisional left-to-right assignments, "
            "not confirmed SAM-native object identities."
        )
    }


# =========================================
# 4. EXTRACT ORIGINAL VIDEO FRAME
# =========================================

def extract_frame(cap, frame_number):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Could not extract frame {frame_number}"
        )

    return frame


# =========================================
# 5. ENCODE FRAME FOR MUSE SPARK
# =========================================

def encode_frame(frame, output_path):

    # Save an evidence copy at original resolution.
    success = cv2.imwrite(
        str(output_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        raise RuntimeError(
            f"Could not save frame: {output_path}"
        )

    # Encode the same original frame for the API.
    encoded_success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not encoded_success:
        raise RuntimeError("Could not encode video frame.")

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


# =========================================
# 6. BUILD MUSE SPARK PROMPT
# =========================================

def build_prompt(
    frame_number,
    timestamp,
    detection_context
):

    context_json = json.dumps(
        detection_context,
        indent=2
    )

    return f"""
You are the visual evidence reviewer for SOPProof,
an AI-assisted workplace safety inspection prototype.

You are reviewing one ORIGINAL frame extracted from a video.

VIDEO EVIDENCE:
Frame number: {frame_number}
Timestamp: {timestamp:.3f} seconds.

SAM 3.1 DETECTION AND ASSOCIATION CONTEXT:

{context_json}

YOUR TASK:

For every detected person:

1. Inspect the original frame visually.
2. Check whether the person's head is visible.
3. Check whether a safety helmet is visibly worn on that person's head.
4. Use SAM bounding boxes only as supporting evidence.
5. Check for false or unrelated helmet detections.
6. Do not assume that a helmet candidate is being worn merely
   because its box overlaps a person's head region.

CLASSIFICATIONS:

helmet_visible:
A safety helmet is visibly worn on the person's head.

potential_missing_helmet:
The person's head is sufficiently visible, but no helmet
is visibly worn on it.

uncertain:
The head is obscured, cropped, blurry, or visual evidence
is insufficient.

IMPORTANT RULES:

- Examine the original frame, not just coordinates.
- A missing SAM detection is not proof of a violation.
- Do not invent workers or evidence.
- Do not infer legal non-compliance.
- Require human review for potential concerns and uncertain cases.
- Use the existing structured inspection schema.
- Return only the requested structured assessment.

IMPORTANT VIDEO RULE:

This is a sampled frame, not continuous video evidence.
Only assess what is visible in this particular frame.
Do not infer what happened between timestamps.

Return the structured assessment in the supplied schema.
"""


# =========================================
# 7. REVIEW ONE FRAME WITH MUSE SPARK
# =========================================

def review_frame(
    client,
    frame,
    frame_number,
    timestamp,
    detection_context,
    image_output_path
):

    image_url = encode_frame(
        frame,
        image_output_path
    )

    prompt = build_prompt(
        frame_number,
        timestamp,
        detection_context
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            response_format=INSPECTION_SCHEMA
        )
    except Exception as exc:
        describe_model_error("Muse Spark video review", MODEL, exc)
        raise

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Muse Spark returned an empty response."
        )

    report = json.loads(content)

    # Apply conservative human-review safeguards.
    people = report.get("people", [])

    needs_review = (
        bool(report.get("human_review_required", True))
        or report.get("overall_result") != "all_helmets_visible"
        or not people
        or any(
            person.get("assessment") != "helmet_visible"
            for person in people
        )
    )

    report["human_review_required"] = needs_review

    return report


# =========================================
# 8. SAVE COMBINED REPORT
# =========================================

def save_report(report):

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    lines = [
        "SOPPROOF VIDEO INSPECTION REPORT",
        "=" * 45,
        f"Source video: {report['source_video']}",
        f"Duration: {report['duration_seconds']} seconds",
        f"Sampled frames: {len(report['frames'])}",
        "",
        "FRAME-BY-FRAME VISUAL REVIEW",
        "=" * 45,
        ""
    ]

    for item in report["frames"]:

        lines.append(
            f"Frame {item['frame']} | "
            f"Timestamp: {item['timestamp_seconds']} seconds"
        )

        if item.get("error"):

            lines.append("Review error: " + item["error"])

        else:

            result = item["assessment"]

            lines.append(
                "Overall result: "
                + str(result.get("overall_result"))
            )

            lines.append(
                "Human review required: "
                + str(result.get("human_review_required"))
            )

            for index, person in enumerate(
                result.get("people", []),
                start=1
            ):

                lines.append(
                    f"Person {index}: "
                    f"{person.get('assessment', 'unknown')}"
                )

                if person.get("evidence"):
                    lines.append(
                        "Evidence: " + str(person["evidence"])
                    )

        lines.append("")

    lines.extend([
        "=" * 45,
        "FINAL INTERPRETATION",
        report["overall_assessment"],
        "",
        "Human review required: "
        + str(report["human_review_required"]),
        "",
        "LIMITATIONS:",
        *report["limitations"]
    ])

    OUTPUT_TEXT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# =========================================
# 9. MAIN VIDEO INSPECTION
# =========================================

def inspect_video():

    print("\n========== SOPPROOF VIDEO INSPECTION ==========")

    required_files = [
        VIDEO_PATH,
        PERSON_CSV,
        HELMET_CSV,
        ASSOCIATION_CSV
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    OUTPUT_DIR.mkdir(exist_ok=True)
    FRAME_DIR.mkdir(exist_ok=True)

    # Load previously generated detections.
    people = load_csv(PERSON_CSV)
    helmets = load_csv(HELMET_CSV)
    associations = load_csv(ASSOCIATION_CSV)

    people_by_frame = group_by_frame(people)
    helmets_by_frame = group_by_frame(helmets)
    associations_by_frame = group_by_frame(associations)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open source video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Some containers report no frame count: count actual frames instead
    # of assuming metadata (or a fixed frame number) is correct.
    if total_frames <= 0:
        counted = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.grab():
            counted += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = counted

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video information.")

    duration = total_frames / fps

    print("Video FPS:", fps)
    print("Total frames:", total_frames)
    print("Duration:", round(duration, 3), "seconds")

    # Select exactly five frames spanning the full duration.
    sample_frames = select_sample_frames(total_frames)

    print("\nFrames selected for Muse Spark review:")

    for frame_number in sample_frames:
        print(
            f"Frame {frame_number} | "
            f"{frame_number / fps:.3f} seconds"
        )

    print("\nInitializing Muse Spark...")

    client = get_meta_client(timeout=180.0)

    result = {
        "project": "SOPProof",
        "source_video": VIDEO_PATH.name,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": round(duration, 3),
        "review_method": "Selected original-frame visual review",
        "sampled_frame_count": len(sample_frames),
        "frames": [],
        "human_review_required": True,
        "overall_assessment": "Review pending",
        "limitations": [
            "Only selected video frames were reviewed by Muse Spark.",
            "Unreviewed intervals cannot be assumed compliant.",
            "Worker track identities are provisional.",
            "SAM geometric associations are supporting evidence, not proof of helmet use.",
            "Potential concerns and uncertain cases require human verification.",
            "This prototype is not a certified safety audit or legal determination."
        ]
    }

    try:

        for index, frame_number in enumerate(
            sample_frames,
            start=1
        ):

            timestamp = frame_number / fps

            print(
                f"\nReviewing sample {index}/{len(sample_frames)}..."
            )

            frame_record = {
                "frame": frame_number,
                "timestamp_seconds": round(timestamp, 3),
                "image": (
                    f"video_review_frames/"
                    f"frame_{frame_number:04d}.jpg"
                )
            }

            try:

                frame = extract_frame(
                    cap,
                    frame_number
                )

                detection_context = prepare_frame_context(
                    frame_number,
                    people_by_frame,
                    helmets_by_frame,
                    associations_by_frame
                )

                image_path = (
                    FRAME_DIR
                    / f"frame_{frame_number:04d}.jpg"
                )

                assessment = review_frame(
                    client,
                    frame,
                    frame_number,
                    timestamp,
                    detection_context,
                    image_path
                )

                frame_record["assessment"] = assessment

                print(
                    "Overall result:",
                    assessment.get("overall_result")
                )

                print(
                    "Human review required:",
                    assessment["human_review_required"]
                )

            except Exception as error:

                frame_record["error"] = (
                    type(error).__name__ + ": " + str(error)
                )

                print(
                    "FRAME REVIEW FAILED:",
                    frame_record["error"]
                )

            result["frames"].append(frame_record)

            # Save after every frame so partial work is retained.
            save_report(result)

    finally:
        cap.release()

    successful_reviews = [
        item for item in result["frames"]
        if "assessment" in item
    ]

    results = [
        item["assessment"]
        for item in successful_reviews
    ]

    has_potential_concern = any(
        assessment.get("overall_result")
        == "potential_missing_helmet"
        or any(
            person.get("assessment")
            == "potential_missing_helmet"
            for person in assessment.get("people", [])
        )
        for assessment in results
    )

    has_uncertain_result = any(
        assessment.get("overall_result") == "uncertain"
        or any(
            person.get("assessment") == "uncertain"
            for person in assessment.get("people", [])
        )
        for assessment in results
    )

    if has_potential_concern:

        result["overall_assessment"] = (
            "Potential missing helmet detected in one or more "
            "sampled frames. Human review required."
        )

    elif has_uncertain_result:

        result["overall_assessment"] = (
            "Uncertain visual evidence in one or more sampled frames."
        )

    elif successful_reviews:

        result["overall_assessment"] = (
            "No potential missing helmet identified in the reviewed "
            "sampled frames. Unreviewed video intervals remain unverified."
        )

    else:

        result["overall_assessment"] = (
            "Video review could not be completed successfully."
        )

    # Always require human review because this is sampled video analysis.
    result["human_review_required"] = True

    result["successful_frame_reviews"] = len(successful_reviews)
    result["failed_frame_reviews"] = (
        len(result["frames"]) - len(successful_reviews)
    )

    save_report(result)

    print("\n========== VIDEO INSPECTION COMPLETED ==========")
    print("Successful reviews:", len(successful_reviews))
    print("Failed reviews:", result["failed_frame_reviews"])
    print("Overall assessment:", result["overall_assessment"])
    print("Human review required: True")
    print("JSON report:", OUTPUT_JSON)
    print("Timeline:", OUTPUT_TEXT)


if __name__ == "__main__":
    inspect_video()

