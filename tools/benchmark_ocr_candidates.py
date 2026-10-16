"""Compare small Paddle models on the same independent local truth, on CPU.

Windows host (uses the existing Docker container, without rebuilding it):
    python tools/benchmark_ocr_candidates.py --container tiktok-translator --allow-downloads

Linux/container: specify --dataset, --media-root and --output. Each candidate
runs in a fresh process so its peak RSS does not include previous candidates.
No original post, production settings or dependencies are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
sys.path.insert(0, str(PROJECT / "translator"))

from regression_metrics import fixture_failures, load_dataset, media_root, ocr_score, time_summary

DOCUMENTATION = "https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html"
MODELS = {
    "pp5_mobile": {"detector": "PP-OCRv5_mobile_det", "recognizer": "PP-OCRv5_mobile_rec",
                   "ocr_version": "PP-OCRv5", "documented_weight_mb": 20.7},
    "pp5_english": {"detector": "PP-OCRv5_mobile_det", "recognizer": "en_PP-OCRv5_mobile_rec",
                    "ocr_version": "PP-OCRv5", "documented_weight_mb": 12.2},
    "pp6_tiny": {"detector": "PP-OCRv6_tiny_det", "recognizer": "PP-OCRv6_tiny_rec",
                 "ocr_version": "PP-OCRv6", "documented_weight_mb": 6.3},
    "pp6_small": {"detector": "PP-OCRv6_small_det", "recognizer": "PP-OCRv6_small_rec",
                  "ocr_version": "PP-OCRv6", "documented_weight_mb": 30.0},
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rss_mb():
    """Actual Linux RSS and high-water mark; not model weight estimates."""
    import resource
    current = None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = round(int(line.split()[1]) / 1024, 3)
                break
    return {"current_rss_mb": current,
            "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)}


def selected_cases(dataset, identifiers):
    if identifiers:
        unknown = set(identifiers) - set(dataset["slides"])
        if unknown:
            raise ValueError("Unknown case IDs: " + ", ".join(sorted(unknown)))
        return {key: case for key, case in dataset["slides"].items() if key in identifiers}
    return dataset["slides"]


def summarize(rows, expected_count):
    measured = [row for row in rows if "metrics" in row]
    return {
        "expected_slides": expected_count,
        "measured_slides": len(measured),
        "errors": len(rows) - len(measured),
        "detection_slides": sum(row["metrics"]["all_regions_detected"] for row in measured),
        "detection_regions": sum(row["metrics"]["detected_regions"] for row in measured),
        "expected_regions": sum(row.get("expected_regions", row.get("metrics", {}).get("expected_regions", 0))
                                for row in rows),
        "measured_expected_regions": sum(row["metrics"]["expected_regions"] for row in measured),
        "exact_recognition_slides": sum(row["metrics"]["recognition"]["exact_normalised"] for row in measured),
        "empty_visible_failures": sum(row["metrics"]["visible_text_missing"] for row in measured),
        "cer": (sum(row["metrics"]["recognition"]["character_errors"] for row in measured)
                / max(1, sum(row["metrics"]["recognition"]["expected_characters"] for row in measured)))
               if measured else None,
        "inference_time": time_summary(row["seconds"] for row in measured),
    }


def run_worker(args, dataset, root, cases):
    specification = MODELS[args.worker]
    cache = Path.home() / ".paddlex" / "official_models"
    missing_models = [name for name in (specification["detector"], specification["recognizer"])
                      if not (cache / name / "inference.json").is_file()
                      and not (cache / name / "inference.pdmodel").is_file()]
    entry = {"candidate": args.worker, "configuration": specification,
             "cpu_threads": args.cpu_threads, "cases": [],
             "models_missing_from_cache_before_run": missing_models,
             "initialisation_includes_download": bool(missing_models),
             "documentation": DOCUMENTATION}
    if missing_models and not args.allow_downloads:
        entry.update(status="ERROR", error="MODEL_NOT_CACHED_REQUIRES_ALLOW_DOWNLOADS")
        save(args.output, entry)
        return 1
    if missing_models:
        print("Explicitly allowed small model downloads: " + ", ".join(missing_models), flush=True)
    initialisation = time.perf_counter()
    try:
        from paddleocr import PaddleOCR
        from src.image.cleaning import read_image
        from src.ocr.paddle import parse_results

        entry["runtime"] = {"opencv": __import__("cv2").__version__,
                            "paddleocr": importlib.metadata.version("paddleocr"),
                            "paddlepaddle": importlib.metadata.version("paddlepaddle")}
        entry["rss_before_model"] = rss_mb()
        engine = PaddleOCR(lang="en", ocr_version=specification["ocr_version"],
                           text_detection_model_name=specification["detector"],
                           text_recognition_model_name=specification["recognizer"],
                           use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False, device="cpu", enable_mkldnn=True,
                           cpu_threads=args.cpu_threads)
        entry["initialisation_seconds"] = round(time.perf_counter() - initialisation, 3)
        entry["rss_after_initialisation"] = rss_mb()
    except Exception as exc:
        entry.update(status="ERROR", error="MODEL_INITIALISATION_ERROR",
                     detail=str(exc), traceback=traceback.format_exc(),
                     initialisation_seconds=round(time.perf_counter() - initialisation, 3))
        save(args.output, entry)
        return 1
    for identifier, case in cases.items():
        row = {"case_id": identifier, "expected_regions": len(case.get("expected_regions", []))}
        try:
            image = read_image(root / case["image"])
            started = time.perf_counter()
            ocr = parse_results(list(engine.predict(image)))
            row.update(seconds=round(time.perf_counter() - started, 3), ocr=ocr,
                       metrics=ocr_score(case, ocr, image.shape), memory=rss_mb())
            print(args.worker, identifier, row["seconds"],
                  "detection", row["metrics"]["all_regions_detected"],
                  "CER", round(row["metrics"]["recognition"]["cer"], 4), flush=True)
        except Exception as exc:
            row.update(error="OCR_ERROR", detail=str(exc))
        entry["cases"].append(row)
        save(args.output, entry)
    entry["summary"] = summarize(entry["cases"], len(cases))
    entry.update(status="MEASURED" if not entry["summary"]["errors"] else "ERROR", memory=rss_mb())
    save(args.output, entry)
    return 0 if entry["status"] == "MEASURED" else 1


def docker_host(args, dataset, root, cases):
    """Copy only source/tooling and text labels; media uses the existing mount."""
    missing = fixture_failures({"slides": cases}, root)
    if missing:
        save(args.output, {"status": "FAIL", "missing_fixtures": missing, "expected_cases": len(cases)})
        return 1
    remote = "/tmp/tiktok-ocr-candidates-" + uuid.uuid4().hex
    remote_output = remote + "/report.json"
    log = args.output.with_suffix(".docker.log")

    def command(arguments, timeout=60):
        completed = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout)
        with log.open("a", encoding="utf-8") as output:
            output.write(completed.stdout + "\n" + completed.stderr + "\n")
        if completed.returncode:
            raise RuntimeError(f"{arguments[0]} exited {completed.returncode}: "
                               + (completed.stderr or completed.stdout)[-2000:])
        return completed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        command(["docker", "exec", args.container, "mkdir", "-p", remote + "/tools", remote + "/tests/regression"])
        for local, destination in (
            (Path(__file__).resolve(), remote + "/tools/benchmark_ocr_candidates.py"),
            (PROJECT / "tools/regression_metrics.py", remote + "/tools/regression_metrics.py"),
            (args.dataset.resolve(), remote + "/tests/regression/expected.json"),
        ):
            command(["docker", "cp", str(local), args.container + ":" + destination])
        arguments = ["docker", "exec", "-e", "PYTHONPATH=/app:" + remote + "/tools",
                     args.container, "python", "-u", remote + "/tools/benchmark_ocr_candidates.py",
                     "--dataset", remote + "/tests/regression/expected.json", "--media-root", args.container_media_root,
                     "--output", remote_output, "--cpu-threads", str(args.cpu_threads),
                     "--timeout-seconds", str(args.timeout_seconds), "--models", *args.models]
        if args.allow_downloads:
            arguments.append("--allow-downloads")
        for identifier in args.case or []:
            arguments.extend(["--case", identifier])
        print("Running isolated CPU OCR candidates in existing container; no Docker build.", flush=True)
        completed = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=len(args.models) * (args.timeout_seconds + 15) + 60)
        with log.open("a", encoding="utf-8") as output:
            output.write(completed.stdout + "\n" + completed.stderr)
        command(["docker", "cp", args.container + ":" + remote_output, str(args.output)])
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report["transport"] = {"container": args.container, "workdir": remote,
                               "log": log.name, "production_modified": False}
        save(args.output, report)
        for model in report.get("candidates", []):
            print(json.dumps({key: model.get(key) for key in ("candidate", "status", "error", "summary", "memory")}), flush=True)
        return completed.returncode
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        save(args.output, {"status": "FAIL", "error": "DOCKER_TRANSPORT_ERROR", "detail": str(exc),
                           "container_workdir": remote, "log": log.name})
        print(str(exc), flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "tests/regression/expected.json")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / ".cache/regression/ocr-candidates.json")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--case", action="append")
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("OCR_CPU_THREADS", "8")))
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--container", help="Run in an existing Docker container from the host.")
    parser.add_argument("--container-media-root", default="/data/tiktok")
    parser.add_argument("--worker", choices=list(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 32 or args.timeout_seconds <= 0:
        parser.error("CPU threads must be 1..32 and timeout must be positive.")
    dataset = load_dataset(args.dataset)
    root = media_root(dataset, args.media_root)
    try:
        cases = selected_cases(dataset, args.case)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("No cases selected.")
    output = args.output.resolve()
    if output.is_relative_to(root):
        relative = output.relative_to(root)
        if relative.parts and relative.parts[0].isdecimal():
            parser.error("Reports must be outside original post directories.")
    if args.container:
        return docker_host(args, dataset, root, cases)
    missing = fixture_failures({"slides": cases}, root)
    if missing:
        save(args.output, {"status": "FAIL", "missing_fixtures": missing, "expected_cases": len(cases)})
        return 1
    if not sys.platform.startswith("linux"):
        parser.error("Use --container tiktok-translator on Windows; RSS measurement requires Linux.")
    if args.worker:
        return run_worker(args, dataset, root, cases)
    workers = args.output.parent / (args.output.stem + "-workers")
    workers.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "cpu_only": True, "single_pass": True,
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                          "paddleocr": importlib.metadata.version("paddleocr"),
                          "paddlepaddle": importlib.metadata.version("paddlepaddle"),
                          "opencv": __import__("cv2").__version__},
              "documentation": DOCUMENTATION, "candidates": [],
              "metric_limitations": ["Manual approximate geometry measures detection coverage.",
                                     "Normalized CER ignores punctuation, case and spaces.",
                                     "Each candidate runs once in a fresh process; model loading/download is separate.",
                                     "RSS is the worker process high-water mark, not total machine memory.",
                                     "Candidate failure stays in the report; it is not a zero-quality measurement."]}
    for candidate in dict.fromkeys(args.models):
        worker_output = workers / (candidate + ".json")
        arguments = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", candidate,
                     "--dataset", str(args.dataset), "--media-root", str(root),
                     "--output", str(worker_output), "--cpu-threads", str(args.cpu_threads)]
        if args.allow_downloads:
            arguments.append("--allow-downloads")
        for identifier in args.case or []:
            arguments.extend(["--case", identifier])
        print("Starting OCR candidate: " + candidate, flush=True)
        try:
            completed = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=args.timeout_seconds)
            (workers / (candidate + ".log")).write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
            if worker_output.is_file():
                entry = json.loads(worker_output.read_text(encoding="utf-8"))
                if completed.returncode and not entry.get("error"):
                    entry.update(status="ERROR", error="WORKER_EXIT_ERROR", exit_code=completed.returncode)
            else:
                entry = {"candidate": candidate, "status": "ERROR", "error": "WORKER_FAILED_WITHOUT_REPORT",
                         "detail": completed.stderr[-2000:], "exit_code": completed.returncode}
        except subprocess.TimeoutExpired as exc:
            entry = {"candidate": candidate, "status": "ERROR", "error": "WORKER_TIMEOUT",
                     "timeout_seconds": args.timeout_seconds}
            (workers / (candidate + ".log")).write_text(str(exc.stderr or ""), encoding="utf-8")
        report["candidates"].append(entry)
        save(args.output, report)
        print(json.dumps({key: entry.get(key) for key in ("candidate", "status", "error", "summary", "memory")}), flush=True)
    report["status"] = "MEASURED" if all(entry.get("status") == "MEASURED" for entry in report["candidates"]) else "PARTIAL"
    save(args.output, report)
    return 0 if report["status"] == "MEASURED" else 1


if __name__ == "__main__":
    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
