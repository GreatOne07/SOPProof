
import csv
import re
from pathlib import Path
from collections import Counter

import cv2


# =========================================
# 1. CONFIGURATION
# =========================================

VIDEO_PATH = Path("hardhat_video.mp4")

PERSON_RAW = Path("sam_video_person_raw.txt")
HELMET_RAW = Path("sam_video_raw.txt")

OUTPUT_DIR = Path("outputs")

PERSON_CSV = OUTPUT_DIR / "video_person_detections.csv"
HELMET_CSV = OUTPUT_DIR / "video_helmet_detections.csv"
ASSOCIATION_CSV = OUTPUT_DIR / "video_worker_helmet_association.csv"


# Raw SAM bounding-box token
BOX_PATTERN = re.compile(
    r"<\|box;"
    r"x1=(\d+);"
    r"y1=(\d+);"
    r"x2=(\d+);"
    r"y2=(\d+);"
    r"w=(\d+);"
    r"h=(\d+);"
    r"c=([0-9.]+)"
    r"\|>"
)

# Standalone frame marker
FRAME_PATTERN = re.compile(r"<(\d+)f>")


# =========================================
# 2. VIDEO INFORMATION
# =========================================

def get_video_information():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open source video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }


# =========================================
# 3. PARSE RAW SAM VIDEO RESPONSE
# =========================================

def parse_sam_video(raw_path):

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw = raw_path.read_text(encoding="utf-8")

    markers = list(FRAME_PATTERN.finditer(raw))

    if not markers:
        raise RuntimeError(
            f"No frame markers found in {raw_path}"
        )

    frames = {}

    for index, marker in enumerate(markers):

        frame_number = int(marker.group(1))

        start = marker.end()

        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(raw)
        )

        section = raw[start:end]

        matches = BOX_PATTERN.findall(section)

        detections = []

        for detection_index, match in enumerate(matches, start=1):

            x1, y1, x2, y2, width, height, confidence = match

            detection = {
                "detection_index": detection_index,
                "left": int(x1),
                "top": int(y1),
                "right": int(x2),
                "bottom": int(y2),
                "frame_width": int(width),
                "frame_height": int(height),
                "confidence": float(confidence)
            }

            detections.append(detection)

        if frame_number in frames:
            raise RuntimeError(
                f"Duplicate frame marker: {frame_number}"
            )

        frames[frame_number] = detections

    print(f"Parsed {raw_path.name}")
    print("Frames:", len(frames))
    print(
        "Bounding boxes:",
        sum(len(items) for items in frames.values())
    )

    return frames


# =========================================
# 4. ASSIGN PROVISIONAL TRACK IDENTITIES
# =========================================

def assign_provisional_tracks(frames, prefix):

    tracked_frames = {}

    for frame_number, detections in frames.items():

        # Sort by horizontal center, left to right
        ordered = sorted(
            detections,
            key=lambda d: (
                d["left"] + d["right"]
            ) / 2
        )

        tracked = []

        for index, detection in enumerate(ordered, start=1):

            item = detection.copy()

            item["track_id"] = f"{prefix}_{index}"

            center_x = (
                item["left"] + item["right"]
            ) / 2

            center_y = (
                item["top"] + item["bottom"]
            ) / 2

            item["center_x"] = center_x
            item["center_y"] = center_y

            tracked.append(item)

        tracked_frames[frame_number] = tracked

    return tracked_frames


# =========================================
# 5. EXPORT DETECTION CSV
# =========================================

def export_detections(frames, output_path, fps):

    columns = [
        "frame",
        "timestamp_seconds",
        "detection_index",
        "track_id",
        "left",
        "top",
        "right",
        "bottom",
        "frame_width",
        "frame_height",
        "confidence"
    ]

    count = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        for frame_number in sorted(frames):

            for detection in frames[frame_number]:

                row = {
                    key: detection[key]
                    for key in columns
                    if key in detection
                }

                row["frame"] = frame_number
                row["timestamp_seconds"] = round(
                    frame_number / fps, 3
                )

                writer.writerow(row)
                count += 1

    print(f"Saved {count} rows to {output_path}")


# =========================================
# 6. WORKER-TO-HELMET ASSOCIATION
# =========================================

def associate_frame(people, helmets):

    candidates = []

    for person_index, person in enumerate(people):

        person_width = person["right"] - person["left"]
        person_height = person["bottom"] - person["top"]

        if person_width <= 0 or person_height <= 0:
            continue

        head_bottom = (
            person["top"] + 0.40 * person_height
        )

        head_center_x = (
            person["left"] + person["right"]
        ) / 2

        head_center_y = (
            person["top"] + head_bottom
        ) / 2

        for helmet_index, helmet in enumerate(helmets):

            hx = helmet["center_x"]
            hy = helmet["center_y"]

            # Check whether the helmet center is
            # inside the approximate head region.
            inside_head = (
                person["left"] <= hx <= person["right"]
                and person["top"] - 0.10 * person_height
                <= hy <= head_bottom
            )

            if not inside_head:
                continue

            score = (
                abs(hx - head_center_x) / person_width
                +
                abs(hy - head_center_y)
                / max(head_bottom - person["top"], 1)
            )

            candidates.append(
                (score, person_index, helmet_index)
            )

    # Greedy one-to-one matching
    candidates.sort()

    assignments = {}
    used_helmets = set()

    for score, person_index, helmet_index in candidates:

        if person_index in assignments:
            continue

        if helmet_index in used_helmets:
            continue

        assignments[person_index] = helmet_index
        used_helmets.add(helmet_index)

    results = []

    for person_index, person in enumerate(people):

        helmet_index = assignments.get(person_index)

        if helmet_index is None:

            helmet = None
            status = "no_helmet_candidate_in_head_region"

        else:

            helmet = helmets[helmet_index]
            status = "helmet_candidate_associated"

        results.append({
            "person_track_id": person["track_id"],
            "person_left": person["left"],
            "person_top": person["top"],
            "person_right": person["right"],
            "person_bottom": person["bottom"],
            "person_confidence": person["confidence"],
            "helmet_track_id": (
                helmet["track_id"] if helmet else ""
            ),
            "helmet_confidence": (
                helmet["confidence"] if helmet else ""
            ),
            "association_status": status,
            "visual_verification_required": True
        })

    return results


# =========================================
# 7. CREATE FRAME-BY-FRAME ASSOCIATION CSV
# =========================================

def export_associations(people_frames, helmet_frames, fps):

    columns = [
        "frame",
        "timestamp_seconds",
        "person_track_id",
        "person_left",
        "person_top",
        "person_right",
        "person_bottom",
        "person_confidence",
        "helmet_track_id",
        "helmet_confidence",
        "association_status",
        "visual_verification_required"
    ]

    counts = Counter()
    total_rows = 0
    warning_frames = []

    with ASSOCIATION_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        common_frames = sorted(
            set(people_frames) & set(helmet_frames)
        )

        for frame_number in common_frames:

            people = people_frames[frame_number]
            helmets = helmet_frames[frame_number]

            # Unexpected detection counts require investigation
            if len(people) != 2 or len(helmets) != 2:
                warning_frames.append(frame_number)

            results = associate_frame(
                people,
                helmets
            )

            for result in results:

                result["frame"] = frame_number

                result["timestamp_seconds"] = round(
                    frame_number / fps, 3
                )

                writer.writerow(result)

                counts[result["association_status"]] += 1
                total_rows += 1

    print("\n========== ASSOCIATION SUMMARY ==========")
    print("Frames processed:", len(people_frames))
    print("Worker-frame records:", total_rows)
    print(
        "Helmet candidate associations:",
        counts["helmet_candidate_associated"]
    )
    print(
        "Unmatched worker-frame records:",
        counts["no_helmet_candidate_in_head_region"]
    )

    print("Frames with unexpected detection counts:",
          len(warning_frames))

    if warning_frames:
        print("Warning frames:", warning_frames[:20])

    return counts


# =========================================
# 8. MAIN
# =========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== SOPPROOF VIDEO PIPELINE ==========\n")

    info = get_video_information()

    fps = info["fps"]

    print("Video FPS:", fps)
    print("Video frames:", info["frame_count"])
    print(
        "Resolution:",
        info["width"], "x", info["height"]
    )

    # Parse both saved SAM responses
    helmet_raw_frames = parse_sam_video(HELMET_RAW)
    person_raw_frames = parse_sam_video(PERSON_RAW)

    # Assign provisional identities based on horizontal position
    helmets = assign_provisional_tracks(
        helmet_raw_frames,
        "helmet"
    )

    people = assign_provisional_tracks(
        person_raw_frames,
        "person"
    )

    # Verify frame coverage
    print("\n========== FRAME COVERAGE ==========")

    print("Helmet frames:", len(helmets))
    print("Person frames:", len(people))

    common = set(helmets) & set(people)

    print("Common frames:", len(common))

    if len(common) != info["frame_count"]:
        print(
            "WARNING: SAM frame coverage differs "
            "from the source video."
        )

    # Export person and helmet detections
    export_detections(
        people,
        PERSON_CSV,
        fps
    )

    export_detections(
        helmets,
        HELMET_CSV,
        fps
    )

    # Associate each person with helmet candidates
    export_associations(
        people,
        helmets,
        fps
    )

    print("\n========== PIPELINE COMPLETED ==========")
    print("Person CSV:", PERSON_CSV)
    print("Helmet CSV:", HELMET_CSV)
    print("Association CSV:", ASSOCIATION_CSV)

    print(
        "\nIMPORTANT: Track identities are provisional "
        "left-to-right assignments, not SAM-native IDs."
    )

    print(
        "Geometric helmet associations require "
        "visual verification."
    )


if __name__ == "__main__":
    main()

