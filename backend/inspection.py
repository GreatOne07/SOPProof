
import base64
import json
import mimetypes

from pathlib import Path

from backend.meta_client import claude_model, describe_model_error, get_meta_client
from backend.sam_detector import detect_image, extract_boxes
from backend.association import associate_helmets


MODEL = claude_model()


# Muse Spark structured-output schema
INSPECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sopproof_inspection",
        "schema": {
            "type": "object",
            "properties": {
                "image_summary": {
                    "type": "string"
                },
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "person_number": {
                                "type": "integer"
                            },
                            "head_visibility": {
                                "type": "string",
                                "enum": [
                                    "clear",
                                    "partially_obscured",
                                    "not_visible"
                                ]
                            },
                            "helmet_observation": {
                                "type": "string",
                                "enum": [
                                    "visible_on_head",
                                    "not_visible",
                                    "uncertain"
                                ]
                            },
                            "assessment": {
                                "type": "string",
                                "enum": [
                                    "helmet_visible",
                                    "potential_missing_helmet",
                                    "uncertain"
                                ]
                            },
                            "evidence": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "person_number",
                            "head_visibility",
                            "helmet_observation",
                            "assessment",
                            "evidence"
                        ],
                        "additionalProperties": False
                    }
                },
                "overall_result": {
                    "type": "string",
                    "enum": [
                        "all_helmets_visible",
                        "potential_missing_helmet",
                        "uncertain"
                    ]
                },
                "human_review_required": {
                    "type": "boolean"
                },
                "limitations": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "image_summary",
                "people",
                "overall_result",
                "human_review_required",
                "limitations"
            ],
            "additionalProperties": False
        }
    }
}


def inspect_image(
    image_path,
    person_raw_path=None,
    helmet_raw_path=None
):
    """
    Run a complete image inspection.

    When cached SAM response files are supplied, reuse them.
    Otherwise, request fresh person and helmet detections.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # ----------------------------------
    # 1. Obtain SAM detections
    # ----------------------------------

    using_cache = (
        person_raw_path is not None
        and helmet_raw_path is not None
    )

    if (person_raw_path is None) != (helmet_raw_path is None):
        raise ValueError(
            "Provide both cached SAM files or neither."
        )

    if using_cache:
        print("Using cached SAM results...")

        people = extract_boxes(
            Path(person_raw_path).read_text()
        )

        helmets = extract_boxes(
            Path(helmet_raw_path).read_text()
        )

    else:
        print("Detecting people with SAM 3.1...")

        people_result = detect_image(
            image_path,
            "person"
        )

        print("Detecting helmets with SAM 3.1...")

        helmet_result = detect_image(
            image_path,
            "hard hat"
        )

        people = people_result["detections"]
        helmets = helmet_result["detections"]

    if not people:
        raise RuntimeError(
            "No people detected. Inspection cannot proceed."
        )

    print("People detected:", len(people))
    print("Helmet candidates:", len(helmets))

    # ----------------------------------
    # 2. Associate helmets with workers
    # ----------------------------------

    association = associate_helmets(
        people,
        helmets
    )

    print(
        "Workers matched:",
        association["matched_count"]
    )

    print(
        "Workers requiring review:",
        association["review_count"]
    )

    # ----------------------------------
    # 3. Prepare original image
    # ----------------------------------

    mime_type, _ = mimetypes.guess_type(
        str(image_path)
    )

    if mime_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise ValueError("Unsupported image format.")

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    image_url = f"data:{mime_type};base64,{encoded}"

    # ----------------------------------
    # 4. Prepare evidence for Muse Spark
    # ----------------------------------

    detection_context = {
        "person_detections": people,
        "helmet_detections": helmets,
        "association_results": association
    }

    prompt = f"""
You are the visual evidence reviewer for SOPProof,
an AI-assisted workplace safety inspection prototype.

Inspect the attached ORIGINAL photograph.

SAM 3.1 detection and association data:
{json.dumps(detection_context)}

For every detected person:

1. Inspect the person's original image region.
2. Check whether their head is visible.
3. Check whether a safety helmet is visibly worn
   on that person's head.
4. Use the bounding boxes to help associate helmets.
5. Check for false or unrelated helmet detections.

Classification:

helmet_visible:
A helmet is visibly worn on the worker's head.

potential_missing_helmet:
The head is sufficiently visible, but no helmet
is visibly worn on it.

uncertain:
The head is obscured, cropped, blurry, or the
available evidence is insufficient.

Important rules:

- Examine the original image, not just coordinates.
- A missing SAM detection is not proof of a violation.
- Do not invent workers or evidence.
- Do not infer legal non-compliance.
- Require human review for potential concerns
  and uncertain cases.
- This is not a certified safety audit.

Return only the requested structured assessment.
"""

    # ----------------------------------
    # 5. Ask Muse Spark to review evidence
    # ----------------------------------

    print("\nSending evidence to Muse Spark...")

    client = get_meta_client(timeout=120.0)

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
        describe_model_error("Muse Spark review", MODEL, exc)
        raise

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Muse Spark returned an empty response."
        )

    report = json.loads(content)

    # ----------------------------------
    # 6. Apply human-review safeguards
    # ----------------------------------

    assessments = report["people"]

    needs_review = (
        association["human_review_required"]
        or report["human_review_required"]
        or any(
            item["assessment"] != "helmet_visible"
            for item in assessments
        )
    )

    report["human_review_required"] = needs_review

    # Preserve the actual detector/association results
    report["detection_summary"] = association

    report["source_image"] = image_path.name

    report["limitations"].append(
        "Prototype result based on one image; "
        "human verification is required."
    )

    # ----------------------------------
    # 7. Save report
    # ----------------------------------

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"

    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir /
        f"{image_path.stem}_inspection.json"
    )

    output_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nINSPECTION COMPLETED!")
    print("Overall result:", report["overall_result"])
    print(
        "Human review required:",
        report["human_review_required"]
    )
    print("Report saved:", output_path)

    return report

