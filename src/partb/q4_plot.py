#!/usr/bin/env python3
"""Part-B Q4: collect the time-vs-K measurements and plot the two charts.

This is a ready-to-run workflow. With no arguments it builds nothing (you
must have already run `make` in src/partb), runs q1_cpu, q2_malloc, and
q3_unified at the full sweep, writes the measurements to a CSV, and produces
the two charts the handout asks for:

  partb_without_unified.png : cudaMalloc/cudaMemcpy, all three launch
                              scenarios plus the CPU baseline.
  partb_with_unified.png    : cudaMallocManaged, all three launch scenarios
                              plus the CPU baseline.

Both use a log-log scale because CPU and GPU times differ by orders of
magnitude. Scenario 1 is capped at K <= 10 million by the binaries (one
launch per element is too slow at large K), so its curve is shorter than
scenarios 2 and 3; the plot handles the shorter curve without any manual
editing.

Usage:
    python q4_plot.py                     # run the binaries, write CSV + charts
    python q4_plot.py --csv partb.csv     # replot from an existing CSV, no rerun
    python q4_plot.py --out-dir plots     # choose the output directory

Run it from src/partb (or pass --bindir to point at the built binaries).
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe: no display needed to save PNGs
import matplotlib.pyplot as plt

RESULT_RE = re.compile(r"^RESULT,(.+)$")
FULL_SWEEP = [1, 5, 10, 50, 100]  # K in millions; binaries cap scenario 1 themselves


def parse_result_line(line):
    m = RESULT_RE.match(line.strip())
    if not m:
        return None
    fields = {}
    for kv in m.group(1).split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            fields[k] = v
    return fields


def run_binary(path, args):
    out = subprocess.run([path] + [str(a) for a in args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{os.path.basename(path)} exited with code {out.returncode}; "
                 f"fix the program before plotting.\n{out.stderr[-1000:]}")
    rows = []
    for line in out.stdout.splitlines():
        f = parse_result_line(line)
        if f:
            rows.append(f)
    if not rows:
        sys.exit(f"{os.path.basename(path)} produced no RESULT lines; "
                 "it did not run correctly, so there is nothing to plot.")
    bad = [r for r in rows if str(r.get("correct", "1")) == "0"]
    if bad:
        sys.exit(f"{os.path.basename(path)} reported correct=0 on {len(bad)} run(s); "
                 "the kernel output is wrong, so the plot would be misleading. "
                 "Fix correctness first (run grade_local.py to see which case fails).")
    return rows


def collect(bindir):
    """Run the three binaries at the full sweep and return a list of dict rows.
    q1_cpu is the CPU baseline; q2_malloc and q3_unified each report three
    scenarios per K (scenario 1 only up to its cap, which the binary enforces).
    Any failed exit, empty output, or incorrect run stops the script, so the
    charts are only ever built from complete, correct measurements."""
    rows = []
    for prog in ("q1_cpu", "q2_malloc", "q3_unified"):
        path = os.path.join(bindir, prog)
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Build Part B first: run `make` in src/partb.")
        got = run_binary(path, FULL_SWEEP)
        for r in got:
            r["program"] = prog
            rows.append(r)
    return rows


def write_csv(rows, csv_path):
    cols = ["program", "scenario", "K", "n", "time_ms", "checksum", "correct"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def cpu_curve(rows):
    pts = []
    for r in rows:
        if r.get("program") == "q1_cpu":
            try:
                pts.append((float(r["K"]), float(r["time_ms"])))
            except (KeyError, ValueError):
                pass
    return sorted(pts)


def gpu_curves(rows, program):
    """{scenario: [(K, time_ms), ...]} for one GPU program."""
    out = {1: [], 2: [], 3: []}
    for r in rows:
        if r.get("program") != program:
            continue
        try:
            s = int(float(r["scenario"]))
            out[s].append((float(r["K"]), float(r["time_ms"])))
        except (KeyError, ValueError):
            continue
    return {s: sorted(p) for s, p in out.items()}


SCENARIO_LABEL = {
    1: "scenario 1: one thread per launch",
    2: "scenario 2: 256 threads per launch",
    3: "scenario 3: single launch",
}


def plot_chart(title, gpu, cpu, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in (1, 2, 3):
        pts = gpu.get(s, [])
        if pts:
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], marker="o", label=SCENARIO_LABEL[s])
    if cpu:
        ax.loglog([p[0] for p in cpu], [p[1] for p in cpu], marker="s", linestyle="--", color="black", label="CPU baseline")
    ax.set_xlabel("K (millions of elements)")
    ax.set_ylabel("time (ms), log scale")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bindir", default=".", help="Directory with the built q1_cpu/q2_malloc/q3_unified")
    ap.add_argument("--csv", help="Replot from this CSV instead of running the binaries")
    ap.add_argument("--out-dir", default="plots", help="Directory to write the PNGs and CSV into")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.csv:
        rows = read_csv(args.csv)
    else:
        rows = collect(args.bindir)
        csv_path = os.path.join(args.out_dir, "partb_measurements.csv")
        write_csv(rows, csv_path)
        print(f"Wrote {csv_path}")

    cpu = cpu_curve(rows)
    for program, title, fname in (
        ("q2_malloc", "Part B: explicit cudaMalloc/cudaMemcpy", "partb_without_unified.png"),
        ("q3_unified", "Part B: unified memory (cudaMallocManaged)", "partb_with_unified.png"),
    ):
        out_path = os.path.join(args.out_dir, fname)
        plot_chart(title, gpu_curves(rows, program), cpu, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
