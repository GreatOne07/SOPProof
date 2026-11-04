
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("MODEL_API_KEY")
video_path = Path("hardhat_video.mp4")

if not api_key:
    raise SystemExit("API key missing.")

if not video_path.exists():
    raise SystemExit("Video not found.")

client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=api_key,
    timeout=180.0
)

try:
    print("Uploading video to Meta...")

    with open(video_path, "rb") as f:
        uploaded = client.files.create(
            purpose="user_data",
            file=f
        )

    print("Upload successful!")
    print("File ID:", uploaded.id)
    print("Starting SAM 3.1 tracking...")

    stream = client.responses.create(
        model="sam-3.1",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hard hat"
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

    result = "".join(output)

    Path("sam_video_raw.txt").write_text(result)

    frame_ids = re.findall(r"<(\d+)f>", result)

    print("\nSAM VIDEO ANALYSIS COMPLETED!")
    print("Output length:", len(result))
    print("Frame markers:", len(frame_ids))
    print("Unique frames:", len(set(frame_ids)))
    print("Saved to sam_video_raw.txt")
    print("\nSample response:")
    print(result[:2000])

except Exception as e:
    print("\nVIDEO ANALYSIS FAILED")
    print(type(e).__name__, str(e))
    raise SystemExit(1) from e

