
import os
import csv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    image_segmentation_format
)

# ------------------------------------
# 1. Configuration
# ------------------------------------

load_dotenv()

api_key = os.getenv("MODEL_API_KEY")

if not api_key:
    raise SystemExit("MODEL_API_KEY missing from .env")

video_path = Path("hardhat_video.mp4")

if not video_path.exists():
    raise SystemExit("hardhat_video.mp4 not found.")

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=api_key,
    timeout=600.0
)

# Separate output files: existing helmet data is untouched
raw_path = Path("sam_video_person_raw.txt")
csv_path = Path("video_person_detections.csv")

# ------------------------------------
# 2. Upload workplace video
# ------------------------------------

try:
    print("Uploading video for PERSON tracking...")

    with open(video_path, "rb") as video_file:
        uploaded = client.files.create(
            purpose="user_data",
            file=video_file
        )

    print("Upload successful!")
    print("File ID:", uploaded.id)

    # --------------------------------
    # 3. Request SAM 3.1 person tracking
    # --------------------------------

    print("\nStarting SAM 3.1 PERSON tracking...")
    print("Prompt: person")

    stream = client.responses.create(
        model="sam-3.1",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "person"
                    },
                    {
                        "type": "input_video",
                        "file_id": uploaded.id
                    }
                ]
            }
        ],
        stream=True,
        metadata={
            "include_confidence": "true"
        }
    )

    output = []

    for event in stream:
        if event.type == "response.output_text.delta":
            output.append(event.delta)

    result_text = "".join(output)

    if not result_text:
        raise RuntimeError("SAM returned an empty response.")

    # --------------------------------
    # 4. Save raw SAM output
    # --------------------------------

    raw_path.write_text(
        result_text,
        encoding="utf-8"
    )

    print("\nSAM PERSON TRACKING COMPLETED!")
    print("Response length:", len(result_text))
    print("Saved:", raw_path)

    # --------------------------------
    # 5. Parse video response
    # --------------------------------

    print("\nParsing SAM video response...")

    sam_format = image_segmentation_format()
    parser = sam_format.create_parser()

    parser.push(result_text)

    finished = parser.finish(
        CompletedOutcome()
    )

    parsed_result = finished.result

    boxes = [
        record
        for record in parsed_result.records
        if isinstance(record, SegmentationBoxRecord)
    ]

    print("Total parsed records:", len(parsed_result.records))
    print("Person bounding boxes:", len(boxes))

    # --------------------------------
    # 6. Export person tracking CSV
    # --------------------------------

    rows = []

    for record in boxes:

        frame_reference = getattr(
            record, "frame", None
        )

        frame_index = (
            getattr(frame_reference, "frame_index", None)
            if frame_reference is not None
            else None
        )

        rows.append({
            "frame": frame_index,
            "object_id": record.object_id,
            "confidence": record.confidence,
            "left": record.left,
            "top": record.top,
            "right": record.right,
            "bottom": record.bottom
        })

    if rows:
        fieldnames = [
            "frame",
            "object_id",
            "confidence",
            "left",
            "top",
            "right",
            "bottom"
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

        print("\nPERSON TRACKING CSV SAVED!")
        print("File:", csv_path)

        unique_objects = set(
            row["object_id"] for row in rows
        )

        unique_frames = set(
            row["frame"]
            for row in rows
            if row["frame"] is not None
        )

        print("Unique tracked people:", len(unique_objects))
        print("Unique frames:", len(unique_frames))

        print("\nObject IDs:")
        for object_id in sorted(unique_objects):
            object_rows = [
                row for row in rows
                if row["object_id"] == object_id
            ]

            print(
                f"Object {object_id}: "
                f"{len(object_rows)} detections"
            )

    else:
        print("WARNING: No bounding boxes were parsed.")
        print("Raw response has been saved for inspection.")

    print("\nPERSON VIDEO ANALYSIS FINISHED!")

except Exception as error:
    print("\nPERSON VIDEO ANALYSIS FAILED")
    print(type(error).__name__, str(error))
    raise SystemExit(1) from error

