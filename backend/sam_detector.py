
import base64
import mimetypes
import re

from pathlib import Path

from backend.meta_client import get_meta_client


MODEL = "sam-3.1"

BOX_PATTERN = (
    r"<\|box;x1=(\d+);y1=(\d+);"
    r"x2=(\d+);y2=(\d+);"
    r"w=(\d+);h=(\d+);c=([0-9.]+)\|>"
)


def extract_boxes(raw_text):
    """Extract bounding boxes from SAM's raw output."""

    matches = re.findall(BOX_PATTERN, raw_text)

    detections = []

    for match in matches:
        x1, y1, x2, y2, width, height, confidence = match

        detections.append({
            "left": int(x1),
            "top": int(y1),
            "right": int(x2),
            "bottom": int(y2),
            "image_width": int(width),
            "image_height": int(height),
            "confidence": float(confidence)
        })

    return detections


def detect_image(image_path, prompt):
    """
    Send an image to SAM 3.1 and return raw output
    and extracted bounding-box detections.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    mime_type, _ = mimetypes.guess_type(str(image_path))

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

    client = get_meta_client(timeout=120.0)

    stream = client.responses.create(
        model=MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ],
        stream=True,
        metadata={
            "mask_encoding": "one_bit",
            "include_confidence": "true"
        }
    )

    chunks = []

    for event in stream:
        if event.type == "response.output_text.delta":
            chunks.append(event.delta)

    raw_text = "".join(chunks)

    if not raw_text:
        raise RuntimeError("SAM returned an empty response.")

    detections = extract_boxes(raw_text)

    return {
        "prompt": prompt,
        "model": MODEL,
        "raw_output": raw_text,
        "detections": detections,
        "count": len(detections)
    }

