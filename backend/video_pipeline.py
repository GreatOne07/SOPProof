import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("SOPPROOF_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


def run_step(label, script_path, job_dir, env, required_files):
    print(f"\n========== {label} ==========")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(job_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{label} timed out after 30 minutes."
        ) from exc

    log_path = job_dir / f"{script_path.stem}.log"
    log_path.write_text(
        (result.stdout or "")
        + "\n\n--- STDERR ---\n"
        + (result.stderr or ""),
        encoding="utf-8",
    )

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "")[-3000:]
        raise RuntimeError(f"{label} failed:\n{details}")

    for relative_path in required_files:
        output_path = job_dir / relative_path

        if not output_path.is_file() or output_path.stat().st_size == 0:
            details = (result.stdout or "")[-2000:]
            raise RuntimeError(
                f"{label} did not create {relative_path}.\n{details}"
            )

    print(f"{label} completed.")
    print("Log:", log_path)


def _safe_original_name(original_filename, fallback):
    """Basename-only display name, short enough to store inside a job dir."""
    name = Path(str(original_filename or "")).name.strip()
    if not name or name in (".", ".."):
        return fallback
    if len(name) > 100:
        suffix = Path(name).suffix[:10]
        name = name[: 100 - len(suffix)] + suffix
    return name


def run_video_pipeline(source_video_path, inspection_id, original_filename=None):
    source_video_path = Path(source_video_path).resolve()

    if not source_video_path.is_file():
        raise FileNotFoundError(source_video_path)

    # Only allow safe, generated inspection IDs.
    if not inspection_id or not inspection_id.isalnum():
        raise ValueError("Invalid inspection ID.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Every upload gets its own isolated working directory.
    job_dir = OUTPUT_ROOT / inspection_id
    job_dir.mkdir(parents=True, exist_ok=False)

    # Existing test scripts expect this exact filename.
    # Keep the internal filename required by existing SAM scripts.
    internal_video_path = job_dir / "hardhat_video.mp4"
    shutil.copy2(source_video_path, internal_video_path)

    # Also preserve the original uploaded filename for the report.
    # Display/metadata only: basename-sanitised and length-capped so a
    # hostile or very long name can never escape or break the job dir.
    original_filename = _safe_original_name(
        original_filename, source_video_path.name
    )

    original_copy_path = job_dir / original_filename

    if original_copy_path != internal_video_path:
        shutil.copy2(source_video_path, original_copy_path)

    env = os.environ.copy()
    env["SOPPROOF_ORIGINAL_VIDEO_NAME"] = original_filename
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    env["SOPPROOF_VIDEO_JOB_DIR"] = str(job_dir)

    # Stage 1: SAM helmet detection.
    run_step(
        "SAM 3.1 helmet detection",
        PROJECT_ROOT / "test_sam_video.py",
        job_dir,
        env,
        ["sam_video_raw.txt"],
    )

    # Stage 2: SAM person tracking.
    run_step(
        "SAM 3.1 person tracking",
        PROJECT_ROOT / "test_sam_video_person.py",
        job_dir,
        env,
        ["sam_video_person_raw.txt"],
    )

    # Stage 3: Parse detections and associate workers with helmets.
    run_step(
        "Detection parsing and association",
        PROJECT_ROOT / "build_video_detections.py",
        job_dir,
        env,
        [
            "outputs/video_person_detections.csv",
            "outputs/video_helmet_detections.csv",
            "outputs/video_worker_helmet_association.csv",
        ],
    )

    # Stage 4: Muse Spark frame review and HTML report.
    run_step(
        "Muse Spark review and report generation",
        PROJECT_ROOT / "backend/run_video_review.py",
        job_dir,
        env,
        [
            "outputs/video_inspection.json",
            "outputs/video_inspection_report.html",
        ],
    )

    report_path = (
        job_dir / "outputs" / "video_inspection_report.html"
    )

    print("\n========== VIDEO PIPELINE FINISHED ==========")
    print("Inspection ID:", inspection_id)
    print("Report:", report_path)

    return report_path
