"""
monitor_process_rss.py
=======================

Standalone external memory monitor for macOS.

PURPOSE
-------
Watches a target process (e.g. your Jupyter kernel, identified by PID)
from OUTSIDE its own execution — this script runs as a completely
separate OS process. It has zero code-level coupling to whatever your
target process is doing internally.

This means: your notebook's process_pdf / DocumentConverter / HybridChunker
code does not need a single line changed, imported, or touched. You run
your notebook exactly as you already have it. Separately, in a terminal,
you run this script pointed at your kernel's PID, and it logs RSS
(physical RAM actually used by that process) at regular intervals to
both the console and a CSV file, for as long as it runs.

Afterward, you correlate the CSV's timestamps against your own knowledge
of when you started/finished running the processing cell(s) in the
notebook (e.g. "I hit Run at 11:42:03, it finished around 11:43:35") to
figure out peak RSS during that window.

WHY EXTERNAL MONITORING INSTEAD OF IN-PROCESS CHECKPOINTS
-----------------------------------------------------------
Trade-off, stated explicitly:
  + Zero coupling: your pipeline code is completely untouched.
  + Works for ANY process, not just Python code you control — could
    equally monitor a subprocess, a compiled binary, another language.
  - Less precision: this script has no visibility into what your code
    is doing internally at any given moment. It only sees "process X
    currently holds Y MB of physical RAM" at each sample tick. You must
    supply the "what was happening at that time" context yourself,
    e.g. by noting wall-clock times as you run notebook cells, or by
    having your notebook print a plain timestamp (not a profiler
    checkpoint, just `import time; print(time.time())`) at stage
    boundaries if you want tighter correlation without importing any
    profiling machinery.

USAGE
-----
1. Find your Jupyter kernel's PID (see two methods below).
2. Run this script in a separate terminal, BEFORE you run your
   processing cells in the notebook:

     python monitor_process_rss.py --pid 12345

3. Run your notebook's processing code as normal, in the notebook,
   completely unmodified.
4. When processing finishes, go back to the terminal running this
   script and press Ctrl+C. It will print a summary (baseline RSS,
   peak RSS, when the peak occurred) and leave a full CSV log on disk
   for finer-grained inspection or plotting.

FINDING YOUR KERNEL'S PID (choose one)
----------------------------------------
  Option A (one throwaway line in any notebook cell, not pipeline code):
      import os; print(os.getpid())

  Option B (fully external, zero notebook interaction):
      ps aux -m | grep -i ipykernel
    Take the PID of the kernel matching your notebook (disambiguate by
    the STARTED column if multiple notebooks are running).

OPTIONS
-------
  --pid           (required) PID of the process to monitor
  --interval      seconds between samples (default: 1.0)
  --output        CSV output path (default: rss_monitor_<pid>_<timestamp>.csv)
  --label         optional free-text label stored in the CSV header,
                  useful if you run this multiple times for different
                  experiments (e.g. "sequential_baseline",
                  "4_workers_no_thread_cap")
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional
import psutil


def get_rss_mb(pid: int) -> Optional[float]:
    """
    Query current total RSS (physical RAM used, in MB) for the given PID
    plus all of its recursive children, via psutil. Summing children is
    deliberate: worker processes (e.g. ProcessPoolExecutor) hold the memory
    that matters most for max_workers sizing. Returns None if the parent
    process no longer exists (e.g. kernel was restarted/killed while this
    monitor was running) — the caller decides whether that means "stop
    monitoring" or just "log a gap."
    """
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        return None


def process_exists(pid: int) -> bool:
    """Cheap liveness check distinct from get_rss_mb, for clearer error messages."""
    return psutil.pid_exists(pid)


def count_children(pid: int) -> int:
    """Live child-process count, printed each sample so you can see workers spawn/exit."""
    try:
        return len(psutil.Process(pid).children(recursive=True))
    except psutil.NoSuchProcess:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="External RSS memory monitor — watches a target process "
                    "by PID without requiring any changes to that process's code."
    )
    parser.add_argument("--pid", type=int, required=True,
                         help="PID of the process to monitor (e.g. your Jupyter kernel's PID)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Seconds between samples (default: 1.0)")
    parser.add_argument("--output", type=str, default=None,
                         help="CSV output path (default: auto-generated with PID + timestamp)")
    parser.add_argument("--label", type=str, default="",
                         help="Optional free-text label recorded in the CSV, "
                              "useful for distinguishing multiple monitoring runs")
    args = parser.parse_args()

    if not process_exists(args.pid):
        print(f"ERROR: No process found with PID {args.pid}.")
        print("Double-check the PID via one of:")
        print("  (notebook cell)  import os; print(os.getpid())")
        print("  (terminal)       ps aux -m | grep -i ipykernel")
        sys.exit(1)

    output_path = args.output or f"rss_monitor_{args.pid}_{int(time.time())}.csv"

    print("=" * 70)
    print(f"Monitoring PID {args.pid} — external, zero-coupling RSS sampler")
    if args.label:
        print(f"Label: {args.label}")
    print(f"Sampling every {args.interval}s")
    print(f"Logging to: {output_path}")
    print("Run your notebook's processing code now.")
    print("Press Ctrl+C when processing is finished to see the summary.")
    print("=" * 70)

    samples = []  # list of (elapsed_seconds, wall_clock_iso, rss_mb)
    start_time = time.time()

    csv_file = open(output_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["elapsed_seconds", "wall_clock_time", "rss_mb", "num_children", "label"])
    csv_file.flush()

    try:
        while True:
            rss = get_rss_mb(args.pid)
            elapsed = time.time() - start_time
            wall_clock = datetime.now().isoformat(timespec="seconds")

            if rss is None:
                print(f"[{elapsed:8.1f}s] {wall_clock}  PROCESS ENDED (PID {args.pid} no longer exists)")
                break

            n_children = count_children(args.pid)
            samples.append((elapsed, wall_clock, rss))
            writer.writerow([f"{elapsed:.2f}", wall_clock, f"{rss:.1f}", n_children, args.label])
            csv_file.flush()  # flush every sample so the CSV is safe to read even if interrupted hard

            print(f"[{elapsed:8.1f}s] {wall_clock}  RSS(total) = {rss:8.1f} MB  children={n_children}")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C).")

    finally:
        csv_file.close()

    # --- Summary ---
    if not samples:
        print("No samples were collected — nothing to summarize.")
        return

    baseline = samples[0][2]
    peak_elapsed, peak_wall_clock, peak_rss = max(samples, key=lambda s: s[2])

    print("\n" + "=" * 70)
    print("RSS MONITORING SUMMARY")
    print("=" * 70)
    print(f"PID monitored:                {args.pid}")
    print(f"Total samples:                {len(samples)}")
    print(f"Monitoring duration:          {samples[-1][0]:.1f}s")
    print(f"Baseline RSS (first sample):  {baseline:8.1f} MB")
    print(f"Peak RSS:                     {peak_rss:8.1f} MB")
    print(f"  occurred at:                {peak_elapsed:.1f}s elapsed ({peak_wall_clock})")
    print(f"Net growth (peak - baseline): {peak_rss - baseline:8.1f} MB")
    print(f"Full log saved to:            {output_path}")
    print("=" * 70)
    print(
        "\nNext step: correlate the peak timestamp above against your own\n"
        "notebook's wall-clock timeline (e.g. when you clicked Run, or any\n"
        "plain `print(time.time())` you may have in the notebook already)\n"
        "to identify which pipeline stage was likely responsible.\n"
        "\n"
        "Use this PEAK RSS number to size max_workers for ProcessPoolExecutor:\n"
        "  safe_max_workers ≈ (total_RAM_MB - RAM_used_by_other_apps_MB\n"
        "                      - safety_margin_MB) / peak_RSS_MB\n"
    )


if __name__ == "__main__":
    main()