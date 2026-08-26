#!/usr/bin/env python3
import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


PID_FILE = "processes.pid"

ENGINE_PORT = int(os.getenv("AUTOML_ENGINE_PORT", "8001"))
BASE_URL = f"http://localhost:{ENGINE_PORT}"

SERVICES = {
    "engine": {
        "port": ENGINE_PORT,
        "uvicorn_target": "app.main:app",
        "base_url": BASE_URL,
    },
}

# Test scenarios selecting which curl checks to run against the engine.
SCENARIOS = [
    "webfromfile",
    "webfromurl",
    "im2web",
    "tabular",
    "tabularmvp",
    "visionmvp",
    "multimodal",
    "audio",
    "text",
]

DEFAULT_READY_TIMEOUT_S = 240.0


def run(
    cmd: List[str],
    capture_output: bool = False,
    check: bool = True,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        check=check,
        text=True,
        env=env,
    )


def kill_port(port: int) -> None:
    try:
        proc = run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, check=False)
        pids_text = (proc.stdout or "").strip()
        if not pids_text:
            return
        for pid_str in pids_text.splitlines():
            if not pid_str.strip():
                continue
            try:
                os.kill(int(pid_str.strip()), signal.SIGKILL)
            except Exception:
                pass
    except FileNotFoundError:
        # lsof not available; best-effort skip
        pass


def start_service(name: str) -> subprocess.Popen:
    svc = SERVICES[name]
    cmd = [
        "uv",
        "run",
        "uvicorn",
        svc["uvicorn_target"],
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        str(svc["port"]),
    ]
    # Start detached enough to not block; inherit env and stdio
    proc = subprocess.Popen(cmd)
    return proc


def save_pids(pids: List[int]) -> None:
    with open(PID_FILE, "w", encoding="utf-8") as f:
        for pid in pids:
            f.write(f"{pid}\n")


def cleanup_processes(pids: List[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    # Give them a moment, then force kill if needed
    time.sleep(0.5)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


def wait_for_port(port: int, timeout_seconds: float = 10.0) -> bool:
    # Use curl to probe readiness to avoid adding Python deps
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            cp = run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    "2",
                    f"http://localhost:{port}/openapi.json",
                ],
                capture_output=True,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def test_web() -> None:
    print("=== Testing Website Accessibility ===")
    cmd = [
        "curl",
        "-sN",
        "-X",
        "POST",
        f"{BASE_URL}/automl/automl_plus/web_access/analyze/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "file=@./sample_data/test.html",
    ]
    cp = run(cmd, capture_output=True, check=False)
    # The endpoint streams JSON lines; print raw output
    print(cp.stdout)
    print()


def test_image_to_website() -> None:
    print("=== Testing Image Tools - run_on_image (image + prompt) ===")
    cmd = [
        "curl",
        "-sN",
        "-X",
        "POST",
        f"{BASE_URL}/automl/automl_plus/image_tools/run_on_image_stream/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "prompt=Recreate this image into a website with HTML/CSS/JS and explain how to run it.",
        "-F",
        "image_file=@./sample_data/websample.png",
    ]
    cp = run(cmd, capture_output=True, check=False)
    # Streaming text/plain; print raw streamed output
    print(cp.stdout)


def test_web_url_guidelines() -> None:
    print("=== Testing Website Accessibility (URL + guidelines) ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/automl_plus/web_access/analyze/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "url=https://alfie-project.eu",
        # "url=https://aiod.eu",
        # "-F",
        # "extra_file_input=@./sample_data/wcag_guidelines.txt",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    print("there was data")


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def test_tabular() -> None:
    print("=== Testing AutoML Tabular From AutoDW ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/tabular/best_model/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "user_id=1",
        "-F",
        "dataset_id=4",
        "-F",
        "target_column_name=labels",
        "-F",
        "task_type=tabular_classification",
        "-F",
        "time_budget=10",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    print()


def test_visionmvp() -> None:
    print("=== Testing AutoML Vision - best_model ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/vision/best_model/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "user_id=1",
        "-F",
        "dataset_id=2",
        "-F",
        "filename_column=filename",
        "-F",
        "label_column=label",
        "-F",
        "task_type=image_classification",
        "-F",
        "time_budget=10",
        "-F",
        "model_size=medium",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)

    print()


def test_multimodal() -> None:
    print("=== Testing AutoML Vision - Multimodal ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/vision/multimodal_best_model/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "user_id=1",
        "-F",
        "dataset_id=5",
        "-F",
        "filename_column=image_file_path",
        "-F",
        "label_column=label",
        "-F",
        "time_budget=60",
        "-F",
        "model_size=medium",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    print()


def test_audio() -> None:
    print("=== Testing AutoML Audio - best_model ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/audio/best_model/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "user_id=1",
        "-F",
        "dataset_id=6",
        "-F",
        "filename_column=filename",
        "-F",
        "label_column=label",
        "-F",
        "task_type=audio_classification",
        "-F",
        "time_budget=60",
        "-F",
        "model_size=small",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    print()


def test_text() -> None:
    print("=== Testing AutoML Text - best_model ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        f"{BASE_URL}/automl/text/best_model/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "user_id=1",
        "-F",
        "dataset_id=7",
        "-F",
        "text_column=text",
        "-F",
        "label_column=label",
        "-F",
        "task_type=text_classification",
        "-F",
        "time_budget=60",
        "-F",
        "model_size=small",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and test the ALFIE AutoML engine service (Python replacement for test_services.sh)"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", *SCENARIOS],
        help="Which test scenarios to run against the unified service",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS if args.target == "all" else [args.target]

    # Kill any existing process on the engine port
    kill_port(ENGINE_PORT)

    # Start the combined engine service
    pids: List[int] = []
    try:
        proc = start_service("engine")
        pids.append(proc.pid)

        save_pids(pids)

        def _cleanup():
            print("Stopping servers...")
            cleanup_processes(pids)

        atexit.register(_cleanup)
        signal.signal(signal.SIGINT, lambda sig, frm: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda sig, frm: sys.exit(0))

        # Wait for readiness
        if not wait_for_port(ENGINE_PORT, timeout_seconds=DEFAULT_READY_TIMEOUT_S):
            print(f"Warning: engine on port {ENGINE_PORT} may not be ready.")

        # Run the requested test scenarios
        if "webfromfile" in scenarios:
            test_web()

        if "webfromurl" in scenarios:
            test_web_url_guidelines()

        if "im2web" in scenarios:
            test_image_to_website()

        if "tabular" in scenarios or "tabularmvp" in scenarios:
            test_tabular()

        if "visionmvp" in scenarios:
            test_visionmvp()

        if "multimodal" in scenarios:
            test_multimodal()

        if "audio" in scenarios:
            test_audio()

        if "text" in scenarios:
            test_text()

        print("=== All tests completed ===")
        return 0
    finally:
        # Ensure cleanup happens even if we raised before atexit executes
        cleanup_processes(pids)


if __name__ == "__main__":
    raise SystemExit(main())
