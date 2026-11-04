
import json
import html
from pathlib import Path
from datetime import datetime
from urllib.parse import quote


# =========================================
# 1. PATHS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_JSON = OUTPUT_DIR / "video_inspection.json"
OUTPUT_HTML = OUTPUT_DIR / "video_inspection_report.html"

ANNOTATED_VIDEO = OUTPUT_DIR / "sopproof_video_preview.mp4"


# =========================================
# 2. HTML HELPERS
# =========================================

def esc(value):
    return html.escape(str(value))


def status_class(status):

    if status in ("helmet_visible", "all_helmets_visible"):
        return "success"

    if status == "potential_missing_helmet":
        return "warning"

    return "neutral"


def readable_status(status):
    return str(status).replace("_", " ").title()


def frame_tone(status):
    """Status tone for a reviewed frame, driven only by its stored status."""
    value = str(status or "").lower()
    if value in ("helmet_visible", "all_helmets_visible"):
        return "frame-safe"
    if value in ("uncertain", "potential_missing_helmet", "review_failed", "review failed"):
        return "frame-uncertain"
    return ""


# The synthetic guest demo ships a generic placeholder as its per-worker
# evidence text. When (and only when) that exact placeholder is rendered,
# replace it with a concise explanation derived from the frame's stored
# assessment, labelled as illustrative. Real inspection evidence, which
# never contains this placeholder, is rendered unchanged.
_DEMO_PLACEHOLDER = "DEMO: Synthetic frame assessment for the guest demonstration."
_DEMO_EXPLANATIONS = {
    "helmet_visible": "Worker detected. Hard hat appears visible in the reviewed frame.",
    "all_helmets_visible": "Worker detected. Hard hat appears visible in the reviewed frame.",
    "uncertain": "Hard-hat visibility is uncertain in this frame. Human verification is required.",
}
DEMO_EXPLANATION_LABEL = "Illustrative demo explanation — not live model output."


# =========================================
# 3. RENDER WORKER ASSESSMENTS
# =========================================

def render_person(person, index):

    assessment = person.get("assessment", "uncertain")
    evidence = person.get(
        "evidence",
        "No visual evidence description supplied."
    )

    demo_label = ""
    if str(evidence).strip() == _DEMO_PLACEHOLDER and assessment in _DEMO_EXPLANATIONS:
        evidence = _DEMO_EXPLANATIONS[assessment]
        demo_label = f"""
            <p class="demo-explanation-label">{esc(DEMO_EXPLANATION_LABEL)}</p>"""

    return f"""
    <div class="person-card">

        <div class="person-heading">
            <strong>Person {index}</strong>

            <span class="status {status_class(assessment)}">
                {esc(readable_status(assessment))}
            </span>
        </div>

        <div class="evidence">
            <strong>Visual evidence</strong>
            <p>{esc(evidence)}</p>{demo_label}
        </div>

    </div>
    """


# =========================================
# 4. RENDER REVIEWED FRAMES
# =========================================

def render_frame(frame_record):

    frame_number = int(frame_record.get("frame", 0))
    timestamp = float(
        frame_record.get("timestamp_seconds", 0)
    )

    image_name = f"frame_{frame_number:04d}.jpg"
    image_relative = f"video_review_frames/{image_name}"
    image_path = OUTPUT_DIR / image_relative

    if image_path.exists():

        image_html = f"""
        <img
            class="frame-image"
            src="{esc(image_relative)}"
            alt="Original video frame {frame_number}"
        >
        """

    else:

        image_html = """
        <div class="missing-image">
            Evidence image unavailable
        </div>
        """

    if "error" in frame_record:

        body = f"""
        <div class="error">
            Review failed: {esc(frame_record["error"])}
        </div>
        """

        overall = "Review Failed"

    else:

        assessment = frame_record.get("assessment", {})

        overall = assessment.get(
            "overall_result",
            "uncertain"
        )

        image_summary = assessment.get(
            "image_summary",
            "No image summary supplied."
        )

        people = assessment.get("people", [])

        people_html = "".join(
            render_person(person, index)
            for index, person in enumerate(people, start=1)
        )

        if not people_html:
            people_html = """
            <p class="muted">
                No individual worker assessments supplied.
            </p>
            """

        body = f"""
        <div class="frame-summary">
            <strong>Frame summary</strong>
            <p>{esc(image_summary)}</p>
        </div>

        <h4>Worker assessments</h4>

        {people_html}
        """

    return f"""
    <section class="frame-card {frame_tone(overall)}">

        <div class="frame-header">

            <div>
                <h3>Frame {frame_number}</h3>
                <p>Timestamp: {timestamp:.3f} seconds</p>
            </div>

            <span class="status {status_class(overall)}">
                {esc(readable_status(overall))}
            </span>

        </div>

        {image_html}

        {body}

    </section>
    """


# =========================================
# 5. GENERATE HTML REPORT
# =========================================

def generate_report():

    print("\n========== SOPPROOF VIDEO REPORT ==========")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Inspection JSON not found: {INPUT_JSON}"
        )

    data = json.loads(
        INPUT_JSON.read_text(encoding="utf-8")
    )

    frames = data.get("frames", [])

    successful = data.get(
        "successful_frame_reviews",
        sum("assessment" in frame for frame in frames)
    )

    failed = data.get(
        "failed_frame_reviews",
        len(frames) - successful
    )

    duration = float(
        data.get("duration_seconds", 0)
    )

    total_frames = data.get("total_frames", 0)

    source_name = Path(
        data.get("source_video", "hardhat_video.mp4")
    ).name

    human_review = data.get(
        "human_review_required",
        True
    )

    overall = data.get(
        "overall_assessment",
        "Review completed."
    )

    review_class = "warning" if human_review else "success"

    review_label = "Required" if human_review else "Not required"

    # Poster: the first reviewed frame's evidence image, when it exists
    # beside this report. Relative to the report, like the frame images.
    poster_attr = ""
    if frames:
        try:
            first_frame = int(frames[0].get("frame", 0))
        except (TypeError, ValueError):
            first_frame = 0
        poster_relative = f"video_review_frames/frame_{first_frame:04d}.jpg"
        if (OUTPUT_DIR / poster_relative).exists():
            poster_attr = f' poster="{esc(poster_relative)}"'

    # Original video is in the project root.
    video_path = PROJECT_ROOT / source_name

    if video_path.exists():

        video_html = f"""
        <video controls preload="metadata"{poster_attr}>
            <source src="../{quote(source_name)}" type="video/mp4">
            Your browser does not support video playback.
        </video>
        """

    else:

        video_html = """
        <div class="missing-image">
            Original source video unavailable.
        </div>
        """

    # Annotated video belongs to this job and sits beside this report.
    if ANNOTATED_VIDEO.exists():
        annotated_type = (
            "video/x-msvideo"
            if ANNOTATED_VIDEO.suffix.lower() == ".avi"
            else "video/mp4"
        )
        annotated_html = f"""
        <div class="video-preview">
            <video controls preload="metadata"{poster_attr}
                   style="width:100%;max-height:520px;background:#000;border-radius:12px;">
                <source src="{quote(ANNOTATED_VIDEO.name)}" type="{annotated_type}">
                Your browser does not support video playback.
            </video>
        </div>
        """
    else:
        annotated_html = """
        <span class="muted">Annotated video preview unavailable.</span>
        """

    frames_html = "".join(
        render_frame(frame)
        for frame in frames
    )

    limitations = "".join(
        f"<li>{esc(item)}</li>"
        for item in data.get("limitations", [])
    )

    if not frames_html:
        frames_html = """
        <p class="muted">
            No frame assessments are available.
        </p>
        """

    generated = datetime.now().astimezone().strftime(
        "%d %B %Y, %I:%M %p %Z"
    )

    # =====================================
    # HTML TEMPLATE
    # =====================================

    page = """<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>SOPProof | Video Inspection Report</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 35px 20px;
    background: #f1f5f9;
    color: #172033;
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Arial, sans-serif;
}

.container {
    max-width: 1150px;
    margin: auto;
    background: white;
    padding: 38px;
    border-radius: 18px;
    box-shadow: 0 12px 35px rgba(20, 35, 60, 0.07);
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 30px;
}

.logo {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
}

.subtitle {
    color: #64748b;
    margin-top: 5px;
}

.report-label {
    background: #eff6ff;
    color: #1d4ed8;
    border-radius: 30px;
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.summary {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 25px;
    margin-bottom: 25px;
}

.summary h2 {
    margin-top: 0;
}

.overall {
    font-size: 23px;
    font-weight: 800;
    margin: 15px 0;
}

.status {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 750;
}

.success {
    color: #15803d;
    background: #dcfce7;
}

.warning {
    color: #c2410c;
    background: #fff1e6;
}

.neutral {
    color: #475569;
    background: #e2e8f0;
}

.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 25px 0 35px;
}

.stat {
    padding: 20px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

.stat strong {
    display: block;
    font-size: 28px;
    margin-bottom: 5px;
}

.stat span {
    color: #64748b;
    font-size: 13px;
}

h2 {
    font-size: 22px;
    margin-top: 36px;
}

h3 {
    margin: 0;
}

h4 {
    margin: 20px 0 12px;
}

video {
    width: 100%;
    display: block;
    background: #111827;
    border-radius: 12px;
    margin-top: 15px;
}

.media-link {
    display: inline-block;
    margin-top: 15px;
    color: #1d4ed8;
    font-weight: 650;
    text-decoration: none;
}

.frame-card {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px;
    margin: 20px 0;
}

.frame-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    flex-wrap: wrap;
}

.frame-header p {
    color: #64748b;
    margin: 5px 0 0;
    font-size: 13px;
}

.frame-image {
    width: 100%;
    max-height: 420px;
    object-fit: contain;
    background: #f1f5f9;
    border-radius: 9px;
}

/* Status outline around the frame image only. It never draws or implies
   object-level detection boxes; those remain whatever the source frame
   itself contains. */
.frame-card.frame-safe .frame-image {
    border: 2px solid #46c98d;
}

.frame-card.frame-uncertain .frame-image {
    border: 2px dashed #f0a832;
    outline: 2px dashed #f0a832;
    outline-offset: 3px;
}

.demo-explanation-label {
    color: #92400e;
    font-size: 12px;
    font-weight: 650;
    margin-top: 8px;
}

.frame-summary {
    background: #f8fafc;
    border-radius: 10px;
    padding: 16px;
    margin-top: 18px;
}

.frame-summary p,
.evidence p {
    color: #475569;
    line-height: 1.6;
    margin-bottom: 0;
}

.person-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
}

.person-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.evidence {
    margin-top: 15px;
    padding-top: 13px;
    border-top: 1px solid #e2e8f0;
}

.notice {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    color: #92400e;
    padding: 18px;
    border-radius: 10px;
    margin-top: 25px;
    line-height: 1.6;
}

.limitations {
    color: #64748b;
    line-height: 1.8;
}

.muted {
    color: #64748b;
}

.missing-image {
    background: #f1f5f9;
    color: #64748b;
    padding: 25px;
    text-align: center;
    border-radius: 10px;
}

.error {
    background: #fef2f2;
    color: #b91c1c;
    padding: 15px;
    border-radius: 8px;
}

footer {
    border-top: 1px solid #e2e8f0;
    margin-top: 35px;
    padding-top: 20px;
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
}

@media (max-width: 750px) {

    .container {
        padding: 20px;
    }

    .stats {
        grid-template-columns: repeat(2, 1fr);
    }

    .overall {
        font-size: 19px;
    }

}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <div>
            <div class="logo">SOPProof</div>
            <div class="subtitle">
                AI-Assisted Workplace Safety Inspection
            </div>
        </div>

        <div class="report-label">
            VIDEO INSPECTION REPORT
        </div>

    </div>

    <div class="summary">

        <h2>Inspection Summary</h2>

        <p><strong>Source video:</strong> {{SOURCE}}</p>
        <p><strong>Duration:</strong> {{DURATION}} seconds</p>
        <p><strong>Generated:</strong> {{GENERATED}}</p>

        <p><strong>Overall assessment:</strong></p>

        <div class="overall">{{OVERALL}}</div>

        <span class="status {{REVIEW_CLASS}}">
            Human review: {{REVIEW_LABEL}}
        </span>

    </div>

    <div class="stats">

        <div class="stat">
            <strong>{{TOTAL_FRAMES}}</strong>
            <span>Total video frames</span>
        </div>

        <div class="stat">
            <strong>{{SAMPLED}}</strong>
            <span>Frames sampled</span>
        </div>

        <div class="stat">
            <strong>{{SUCCESSFUL}}</strong>
            <span>Successful AI reviews</span>
        </div>

        <div class="stat">
            <strong>{{FAILED}}</strong>
            <span>Failed AI reviews</span>
        </div>

    </div>

    <h2>Original Workplace Video</h2>

    {{VIDEO}}

    {{ANNOTATED}}

    <h2>Frame-by-Frame AI Review</h2>

    <p class="muted">
        Original video frames selected for visual inspection
        by Meta Muse Spark. Every assessment is tied to its
        original video timestamp.
    </p>

    {{FRAMES}}

    <h2>Limitations</h2>

    <ul class="limitations">
        {{LIMITATIONS}}
    </ul>

    <div class="notice">
        <strong>Important:</strong>
        SOPProof provides prototype, visual evidence-based
        assistance. This report is not a certified safety audit
        or a legal determination. Only selected video frames
        were reviewed by Muse Spark. Unreviewed intervals cannot
        be assumed compliant. Potential concerns require
        human verification.
    </div>

    <footer>
        SOPProof | AI-Assisted Workplace Safety Inspection
    </footer>

</div>

</body>
</html>
"""

    # =====================================
    # INSERT REPORT DATA
    # =====================================

    replacements = {
        "{{SOURCE}}": esc(source_name),
        "{{DURATION}}": f"{duration:.2f}",
        "{{GENERATED}}": esc(generated),
        "{{OVERALL}}": esc(overall),
        "{{REVIEW_CLASS}}": review_class,
        "{{REVIEW_LABEL}}": review_label,
        "{{TOTAL_FRAMES}}": str(total_frames),
        "{{SAMPLED}}": str(len(frames)),
        "{{SUCCESSFUL}}": str(successful),
        "{{FAILED}}": str(failed),
        "{{VIDEO}}": video_html,
        "{{ANNOTATED}}": annotated_html,
        "{{FRAMES}}": frames_html,
        "{{LIMITATIONS}}": limitations
    }

    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)

    OUTPUT_HTML.write_text(
        page,
        encoding="utf-8"
    )

    print("\n========== REPORT GENERATED ==========")
    print("Successful AI reviews:", successful)
    print("Failed AI reviews:", failed)
    print("Human review required:", human_review)
    print("HTML report:", OUTPUT_HTML)


if __name__ == "__main__":
    generate_report()

