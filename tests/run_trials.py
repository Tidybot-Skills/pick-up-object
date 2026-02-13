#!/usr/bin/env python3
"""Run pick-up-object trials and save execution info."""

import sys
import requests
import time
import json
import os
from pathlib import Path

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

API = "http://10.102.253.246:8080"
KEY = "sk-user-224f278fdcb8607a0bc07999564af13ef3c4f05e7e2f4438"
HEADERS = {"X-API-Key": KEY}

SKILL_CODE = Path(__file__).parent.parent / "main.py"
RESULTS_DIR = Path(__file__).parent / "results"


def acquire_lease():
    """Acquire robot lease."""
    r = requests.post(f"{API}/lease/acquire", headers=HEADERS, json={"holder": "jarvis-trials", "rewind_on_release": True})
    r.raise_for_status()
    return r.json()["lease_id"]


def release_lease(lease_id):
    """Release robot lease."""
    requests.post(f"{API}/lease/release", headers=HEADERS, json={"lease_id": lease_id})


def extend_lease(lease_id):
    """Extend lease duration."""
    r = requests.post(f"{API}/lease/extend", headers={**HEADERS, "X-Lease-Id": lease_id}, json={})
    return r.status_code == 200


def run_skill(lease_id):
    """Execute the skill and return execution_id."""
    code = SKILL_CODE.read_text()
    r = requests.post(
        f"{API}/code/execute",
        headers={**HEADERS, "X-Lease-Id": lease_id},
        json={"code": code, "timeout": 180}
    )
    r.raise_for_status()
    return r.json()["execution_id"]


def wait_for_completion(lease_id, timeout=180):
    """Wait for code execution to complete, streaming output."""
    start = time.time()
    stdout_offset = 0
    all_stdout = ""
    
    while time.time() - start < timeout:
        # Extend lease periodically
        if int(time.time() - start) % 60 == 0 and int(time.time() - start) > 0:
            extend_lease(lease_id)
        
        r = requests.get(f"{API}/code/status", headers=HEADERS, params={"stdout_offset": stdout_offset})
        data = r.json()
        
        # Print and capture new stdout
        if data.get("stdout"):
            print(data["stdout"], end="", flush=True)
            all_stdout += data["stdout"]
            stdout_offset += len(data["stdout"])
        
        # Check if still running
        is_running = data.get("is_running", data.get("running", False))
        if not is_running:
            # Get final result
            time.sleep(0.5)
            result_r = requests.get(f"{API}/code/result", headers=HEADERS)
            return result_r.json().get("result", {})
        
        time.sleep(0.5)
    
    print(f"\nWARNING: Timeout after {timeout}s")
    return {"stdout": all_stdout, "timeout": True}


def get_recording(execution_id):
    """Get recording metadata."""
    r = requests.get(f"{API}/code/recordings/{execution_id}", headers=HEADERS)
    if r.status_code == 200:
        return r.json()
    return None


def rewind_to_home(lease_id):
    """Rewind robot to home position."""
    print("Rewinding to home...", flush=True)
    r = requests.post(f"{API}/rewind/reset-to-home", headers={**HEADERS, "X-Lease-Id": lease_id})
    if r.status_code == 200:
        # Wait for rewind to complete
        for _ in range(120):  # Max 2 minutes
            status = requests.get(f"{API}/rewind/status", headers=HEADERS).json()
            if not status.get("is_rewinding", False):
                break
            time.sleep(1)
        print("Rewind complete", flush=True)
    else:
        print(f"Rewind failed: {r.text}", flush=True)


def wait_for_idle():
    """Wait until no code is running."""
    for _ in range(30):
        r = requests.get(f"{API}/code/status", headers=HEADERS)
        data = r.json()
        if not data.get("is_running", data.get("running", False)):
            return True
        time.sleep(1)
    return False


def run_trial(trial_num, lease_id):
    """Run a single trial and return result info."""
    print(f"\n{'='*60}", flush=True)
    print(f"TRIAL {trial_num}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Make sure nothing is running
    wait_for_idle()
    
    # Execute skill
    try:
        exec_id = run_skill(lease_id)
    except requests.exceptions.HTTPError as e:
        print(f"Failed to start execution: {e}", flush=True)
        return {"trial": trial_num, "success": False, "error": str(e)}
    
    print(f"Execution ID: {exec_id}\n", flush=True)
    
    # Wait for completion with streaming output
    result = wait_for_completion(lease_id, timeout=180)
    
    # Determine success
    stdout = result.get("stdout", "")
    success = "Result: SUCCESS" in stdout
    
    # Get recording info
    recording = get_recording(exec_id)
    
    trial_info = {
        "trial": trial_num,
        "execution_id": exec_id,
        "success": success,
        "exit_code": result.get("exit_code"),
        "duration": result.get("duration"),
        "stdout": stdout,
        "stderr": result.get("stderr", ""),
        "recording": recording
    }
    
    # Save trial info
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"trial_{trial_num}.json", "w") as f:
        json.dump(trial_info, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print(f"TRIAL {trial_num} RESULT: {'SUCCESS' if success else 'FAILED'}", flush=True)
    print(f"{'='*60}", flush=True)
    
    return trial_info


def main():
    num_trials = 3
    results = []
    
    # Make sure no code is running
    print("Checking for running code...", flush=True)
    wait_for_idle()
    
    # Acquire lease
    print("Acquiring lease...", flush=True)
    lease_id = acquire_lease()
    print(f"Lease acquired: {lease_id}\n", flush=True)
    
    try:
        for i in range(1, num_trials + 1):
            # Rewind before each trial
            rewind_to_home(lease_id)
            time.sleep(2)  # Brief pause after rewind
            
            # Run trial
            info = run_trial(i, lease_id)
            results.append(info)
            
            # Brief pause between trials
            if i < num_trials:
                print("\nPausing 5s before next trial...", flush=True)
                time.sleep(5)
        
        # Summary
        print(f"\n{'='*60}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)
        successes = sum(1 for r in results if r.get("success", False))
        print(f"Success rate: {successes}/{num_trials} ({100*successes/num_trials:.0f}%)")
        for r in results:
            status = "✓" if r.get("success", False) else "✗"
            exec_id = r.get("execution_id", "N/A")
            print(f"  Trial {r['trial']}: {status} (exec_id: {exec_id})")
        
        # Save summary
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump({
                "num_trials": num_trials,
                "successes": successes,
                "results": [{"trial": r["trial"], "success": r.get("success", False), "execution_id": r.get("execution_id")} for r in results]
            }, f, indent=2)
    
    finally:
        release_lease(lease_id)
        print("\nLease released", flush=True)


if __name__ == "__main__":
    main()
