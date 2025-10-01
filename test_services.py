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

PID_FILE = "processes.pid"


SERVICES = {
    "webfromfile": {
        "port": 8000,
        "uvicorn_target": "app.automlplus.main:app",
        "base_url": "http://localhost:8000",
    },
    "webfromurl": {
        "port": 8000,
        "uvicorn_target": "app.automlplus.main:app",
        "base_url": "http://localhost:8000",
    },
    "im2web": {
        "port": 8000,
        "uvicorn_target": "app.automlplus.main:app",
        "base_url": "http://localhost:8000",
    },
    "tabular": {
        "port": 8001,
        "uvicorn_target": "app.tabular_automl.main:app",
        "base_url": "http://localhost:8001",
    },
    "vision": {
        "port": 8002,
        "uvicorn_target": "app.vision_automl.main:app",
        "base_url": "http://localhost:8002",
    },
}

DEFAULT_READY_TIMEOUT_S = 240.0
GENERAL_READY_TIMEOUT_S = 420.0


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


def wait_for_general_ready(timeout_seconds: float = GENERAL_READY_TIMEOUT_S) -> bool:
    # Wait for /health to report ready=true
    deadline = time.time() + timeout_seconds
    url = "http://localhost:8004/health"
    while time.time() < deadline:
        try:
            cp = run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    "3",
                    url,
                ],
                capture_output=True,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout:
                try:
                    data = json.loads(cp.stdout)
                    if isinstance(data, dict) and bool(data.get("ready")):
                        return True
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        time.sleep(1.0)
    return False


def test_web() -> None:
    print("=== Testing Website Accessibility ===")
    cmd = [
        "curl",
        "-sN",
        "-X",
        "POST",
        "http://localhost:8000/automlplus/web_access/analyze/",
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
        "http://localhost:8000/automlplus/image_tools/run_on_image_stream/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "prompt=Recreate this image into a website with HTML/CSS/JS and explain how to run it.",
        "-F",
        "image_file=@./sample_data/websample.png",
        # Optionally: "-F", "model=qwen2.5vl",
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
        "http://localhost:8000/automlplus/web_access/analyze/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        # "url=https://alfie-project.eu",
        "url=https://aiod.eu",
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
    print("=== Testing AutoML Tabular - get_user_input ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        "http://localhost:8001/automl_tabular/get_user_input/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "train_csv=@./sample_data/knot_theory/train.csv",
        "-F",
        "target_column_name=signature",
        "-F",
        "task_type=classification",
        "-F",
        "time_budget=30",
    ]
    cp = run(cmd, capture_output=True, check=False)
    data = parse_json(cp.stdout or "")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(cp.stdout)
    session_id = data.get("session_id")
    if session_id:
        print("=== Testing AutoML Tabular - find_best_model ===")
        cmd2 = [
            "curl",
            "-s",
            "-X",
            "POST",
            "http://localhost:8001/automl_tabular/find_best_model/",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"session_id": session_id}),
        ]
        cp2 = run(cmd2, capture_output=True, check=False)
        data2 = parse_json(cp2.stdout or "")
        if data2:
            print(json.dumps(data2, indent=2, ensure_ascii=False))
        else:
            print(cp2.stdout)
    else:
        print("Failed to get valid session_id from tabular get_user_input")
    print()


def test_vision() -> None:
    print("=== Testing AutoML Vision - get_user_input ===")
    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        "http://localhost:8002/automl_vision/get_user_input/",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "csv_file=@./sample_data/Garbage_Dataset_Classification/metadata.csv",
        "-F",
        "images_zip=@./sample_data/Garbage_Dataset_Classification/images.zip",
        "-F",
        "filename_column=filename",
        "-F",
        "label_column=label",
        "-F",
        "task_type=classification",
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
    session_id = data.get("session_id")
    if session_id:
        print("=== Testing AutoML Vision - find_best_model ===")
        cmd2 = [
            "curl",
            "-s",
            "-X",
            "POST",
            "http://localhost:8002/automl_vision/find_best_model/",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"session_id": session_id}),
        ]
        cp2 = run(cmd2, capture_output=True, check=False)
        data2 = parse_json(cp2.stdout or "")
        if data2:
            print(json.dumps(data2, indent=2, ensure_ascii=False))
        else:
            print(cp2.stdout)
    else:
        print("Failed to get valid session_id from vision get_user_input")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and test ALFIE services (Python replacement for test_services.sh)"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "webfromfile", "webfromurl", "tabular", "vision", "im2web"],
        help="Which services to run and test",
    )
    args = parser.parse_args()

    targets = [args.target] if args.target != "all" else list(SERVICES.keys())

    # Kill existing processes on expected ports
    for name in targets:
        kill_port(SERVICES[name]["port"])

    # Start services
    procs: Dict[str, subprocess.Popen] = {}
    pids: List[int] = []
    try:
        for name in targets:
            proc = start_service(name)
            procs[name] = proc
            pids.append(proc.pid)
            time.sleep(2)

        save_pids(pids)

        def _cleanup():
            print("Stopping servers...")
            cleanup_processes(pids)

        atexit.register(_cleanup)
        signal.signal(signal.SIGINT, lambda sig, frm: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda sig, frm: sys.exit(0))

        # Wait for readiness
        for name in targets:
            port = SERVICES[name]["port"]
            if name == "general":
                if not wait_for_general_ready(timeout_seconds=GENERAL_READY_TIMEOUT_S):
                    print(
                        f"Warning: Service {name} on port {port} may not be ready (health not ready)."
                    )
                continue
            if not wait_for_port(port, timeout_seconds=DEFAULT_READY_TIMEOUT_S):
                print(f"Warning: Service {name} on port {port} may not be ready.")

        # Run tests mirroring the shell script
        if "webfromfile" in targets:
            test_web()

        if "webfromurl" in targets:
            test_web_url_guidelines()

        if "im2web" in targets:
            test_image_to_website()

        if "tabular" in targets:
            test_tabular()

        if "vision" in targets:
            test_vision()

        print("=== All tests completed ===")
        return 0
    finally:
        # Ensure cleanup happens even if we raised before atexit executes
        cleanup_processes(pids)


if __name__ == "__main__":
    raise SystemExit(main())
