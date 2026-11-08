#!/usr/bin/env python3
"""
grade_local.py -- HPML HW3 (CUDA Kernels) local grader.

Run this on YOUR OWN GPU machine (Insomnia / Colab) after finishing the
code in src/. It builds Part A/B/C, runs the resulting binaries, checks
correctness (exact checksums for A/B, a scaled dot-product attention
reference for C), records timings, stamps host info and a timestamp, and
writes results.json -- the file you commit alongside your code.

Gradescope then (a) compile-checks your code, (b) validates this
results.json against results.schema.json and a few sanity bounds, and
leaves performance analysis and your written report to the TA rubric.

If nvcc or a GPU isn't available, this script automatically falls back to
(or can be forced into with --dry-run) a CPU-only "dry run" that still
produces a schema-valid results.json, with every GPU-only field marked
"skipped: no GPU". Part B's q1_cpu (pure CPU, no CUDA) still actually
builds and runs in this mode. A dry run is useful for testing the harness,
but it does NOT satisfy the assignment -- you need a real GPU run (mode
"gpu" in the resulting results.json) before you submit.

Usage:
    python grade_local.py                # auto-detect GPU, build + run everything
    python grade_local.py --dry-run      # force the CPU-only dry run
    python grade_local.py --arch sm_80   # override the nvcc target architecture
                                          # (T4=sm_75 default, V100=sm_70, A100=sm_80, H100=sm_90)
"""
import argparse
import datetime
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

RESULT_RE = re.compile(r"^RESULT,(.+)$")
INT_RE = re.compile(r"^-?\d+$")


def parse_result_lines(stdout):
    rows = []
    for line in stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if not m:
            continue
        fields = {}
        for kv in m.group(1).split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            fields[k] = v
        rows.append(fields)
    return rows


def coerce_run_row(row):
    """Normalize a parsed RESULT,... dict's string values into ints/floats
    where possible (handles plain integers, decimals, and scientific
    notation like matmult's max_abs_err=1e-05), leaving anything else
    (e.g. program=vecadd00) as a string."""
    out = {}
    for k, v in row.items():
        if k == "_returncode":
            continue
        if not isinstance(v, str):
            # Already a non-string (e.g. the returncode we stamped on): keep it.
            out[k] = v
        elif INT_RE.match(v):
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                out[k] = v
    return out


def run(cmd, cwd=None, timeout=None):
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        return 124, out, err + "\n[grade_local.py] timed out"


def which(name):
    return shutil.which(name)


def detect_host():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "nvcc_version": None,
        "gpu_name": None,
        "driver_version": None,
    }
    if which("nvcc"):
        rc, out, _ = run(["nvcc", "--version"])
        if rc == 0:
            m = re.search(r"release (\S+),", out)
            info["nvcc_version"] = m.group(1).rstrip(",") if m else out.strip().splitlines()[-1]
    if which("nvidia-smi"):
        rc, out, _ = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if rc == 0 and out.strip():
            first = out.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 1 and parts[0]:
                info["gpu_name"] = parts[0]
            if len(parts) >= 2 and parts[1]:
                info["driver_version"] = parts[1]
            info["cuda_available"] = True
    return info


def make_build(target_dir, arch, targets=None, timeout=300):
    """Run `make [targets...] ARCH=arch` in target_dir. Returns (ok, log)."""
    cmd = ["make", f"ARCH={arch}"]
    if targets:
        cmd += list(targets)
    rc, out, err = run(cmd, cwd=target_dir, timeout=timeout)
    return rc == 0, (out or "") + (err or "")


def program_result_skip(reason):
    return {"status": "skipped", "reason": reason, "build_ok": None, "runs": []}


def program_result_fail(reason, build_ok=None, runs=None):
    return {"status": "failed", "reason": reason, "build_ok": build_ok, "runs": runs or []}


def program_result_ok(runs, build_ok=True):
    return {"status": "ok", "reason": None, "build_ok": build_ok, "runs": runs}


def all_runs_correct(rows, expected):
    """A program counts as correct only if it produced EXACTLY the expected
    number of graded runs (runs carrying a `correct` field) AND every one of
    them is correct, AND no run reported a nonzero return code. `expected` is
    the number of graded RESULT lines the program should emit for this
    invocation set. Requiring the full expected count, rather than only
    checking the runs that happen to be present, means a crash or truncated
    output (which drops RESULT lines) fails the check instead of slipping
    through: an any()- or present-only all()-based check would accept a
    program that printed one passing scenario and crashed before the rest."""
    if any(r.get("returncode") not in (None, 0) for r in rows):
        return False
    graded = [r for r in rows if r.get("correct") is not None]
    if len(graded) != expected:
        return False
    return all(r.get("correct") in (1, True) for r in graded)


def run_binary(bin_path, args_list, cwd, timeout=600):
    """Run one binary once per entry of args_list (each entry is itself a
    list of CLI args for that one invocation). One invocation can print
    several RESULT lines (e.g. Part B's internal K-sweep, Part A's single
    line); returns the flattened list of parsed RESULT rows across all
    invocations."""
    all_rows = []
    for args in args_list:
        rc, out, err = run([bin_path] + [str(a) for a in args], cwd=cwd, timeout=timeout)
        rows = parse_result_lines(out)
        if not rows:
            all_rows.append({
                "args": list(args), "returncode": rc, "time_ms": None, "correct": None,
                "stderr_tail": (err or "")[-500:],
            })
        else:
            # Stamp the invocation's exit code onto each parsed row, so a
            # program that printed its RESULT lines but then exited nonzero
            # (e.g. a crash during cleanup) is still caught downstream.
            for r in rows:
                r.setdefault("returncode", rc)
            all_rows.extend(rows)
    return all_rows


def run_binary_stdout(bin_path, args, cwd, timeout=120):
    """Run a binary once and return its raw stdout (used by the Part-C
    element-wise check, which needs the OUT,... dump line, not just the
    parsed RESULT row)."""
    rc, out, _ = run([bin_path] + [str(a) for a in args], cwd=cwd, timeout=timeout)
    # Output printed before a crash or timeout is not a successful run.
    return (out or "") if rc == 0 else ""


def build_and_run_part_a(mode, arch):
    parta_dir = os.path.join(SRC, "parta")
    result = {}
    if mode != "gpu":
        for name in ("vecadd00", "vecadd01", "matmult00", "matmult01"):
            result[name] = program_result_skip("skipped: no GPU (dry-run mode)")
        result["coalescing_speedup"] = None
        result["tiling_speedup"] = None
        return result

    ok, log = make_build(parta_dir, arch)
    if not ok:
        reason = "make failed:\n" + log[-2000:]
        for name in ("vecadd00", "vecadd01", "matmult00", "matmult01"):
            result[name] = program_result_fail(reason, build_ok=False)
        result["coalescing_speedup"] = None
        result["tiling_speedup"] = None
        return result

    # A quick size (fast build/correctness smoke test) and a larger size
    # (a real timing data point) for each binary. matmult00 has
    # FOOTPRINT_SIZE=16, matmult01 has FOOTPRINT_SIZE=32 -- the multiplier
    # arguments below are chosen so both land on the SAME matrix sizes
    # (256x256, 1024x1024), for an apples-to-apples comparison.
    specs = {
        "vecadd00": [[1], [50]],
        "vecadd01": [[1], [50]],
        "matmult00": [[16], [64]],
        "matmult01": [[8], [32]],
    }
    for name, args_list in specs.items():
        bin_path = os.path.join(parta_dir, name)
        if not os.path.exists(bin_path):
            result[name] = program_result_fail("binary not found after build", build_ok=False)
            continue
        rows = [coerce_run_row(r) for r in run_binary(bin_path, args_list, cwd=parta_dir)]
        # One graded RESULT line per invocation (one per entry of args_list).
        ok_all = all_runs_correct(rows, expected=len(args_list))
        result[name] = {
            "status": "ok" if ok_all else "failed",
            "reason": None if ok_all else "at least one run failed its correctness check, crashed, or produced no RESULT line",
            "build_ok": True,
            "runs": rows,
        }

    def time_for(name, key, val):
        for r in result.get(name, {}).get("runs", []):
            if r.get(key) == val:
                return r.get("time_ms")
        return None

    t00 = time_for("vecadd00", "values_per_thread", 50)
    t01 = time_for("vecadd01", "values_per_thread", 50)
    result["coalescing_speedup"] = (t00 / t01) if (t00 and t01) else None

    t_m00 = time_for("matmult00", "size", 1024)
    t_m01 = time_for("matmult01", "size", 1024)
    result["tiling_speedup"] = (t_m00 / t_m01) if (t_m00 and t_m01) else None

    return result


def build_and_run_part_b(mode, arch):
    partb_dir = os.path.join(SRC, "partb")
    result = {}

    # q1_cpu needs only g++, not nvcc/a GPU -- build and run it even in
    # dry-run mode, so the CPU-testable part of the harness gets exercised
    # for real rather than blanket-skipped.
    ok, log = make_build(partb_dir, arch, targets=["q1_cpu"], timeout=120)
    q1_bin = os.path.join(partb_dir, "q1_cpu")
    if not ok or not os.path.exists(q1_bin):
        result["q1_cpu"] = program_result_fail("make q1_cpu failed:\n" + log[-2000:], build_ok=False)
    else:
        rows = [coerce_run_row(r) for r in run_binary(q1_bin, [[1]], cwd=partb_dir, timeout=120)]
        # q1_cpu prints one graded RESULT line for the single K it is run with.
        ok_all = all_runs_correct(rows, expected=1)
        result["q1_cpu"] = {
            "status": "ok" if ok_all else "failed",
            "reason": None if ok_all else "at least one run failed its correctness check, crashed, or produced no RESULT line",
            "build_ok": True,
            "runs": rows,
        }

    if mode != "gpu":
        result["q2_malloc"] = program_result_skip("skipped: no GPU (dry-run mode)")
        result["q3_unified"] = program_result_skip("skipped: no GPU (dry-run mode)")
        return result

    ok, log = make_build(partb_dir, arch, targets=["q2_malloc", "q3_unified"])
    if not ok:
        reason = "make failed:\n" + log[-2000:]
        result["q2_malloc"] = program_result_fail(reason, build_ok=False)
        result["q3_unified"] = program_result_fail(reason, build_ok=False)
        return result

    for name in ("q2_malloc", "q3_unified"):
        bin_path = os.path.join(partb_dir, name)
        if not os.path.exists(bin_path):
            result[name] = program_result_fail("binary not found after build", build_ok=False)
            continue
        # K=1 million only: a fast per-scenario smoke test. Scenario 1 (n
        # sequential single-thread launches) is deliberately slow by
        # design -- that is the point of the exercise -- so the full
        # K=1,5,10,50,100 sweep q4_plot.py expects for the report (run
        # with no CLI args) is left to the student's own run, not this
        # automated grading pass.
        rows = [coerce_run_row(r) for r in run_binary(bin_path, [[1]], cwd=partb_dir, timeout=300)]
        # At K=1 million all three launch scenarios run (scenario 1's K<=10
        # cap includes K=1), so expect three graded RESULT lines.
        ok_all = all_runs_correct(rows, expected=3)
        result[name] = {
            "status": "ok" if ok_all else "failed",
            "reason": None if ok_all else "at least one launch scenario failed its correctness check, crashed, or produced no RESULT line",
            "build_ok": True,
            "runs": rows,
        }

    return result


def reference_attention_output(N, d):
    """Pure-Python reference for the attention output tensor (no NumPy
    dependency), returned as a flat row-major list of N*d doubles. Mirrors
    common.h's makeQ/makeK/makeV and the scaled dot-product attention (safe
    row softmax) used by c1_naive.cu/c2_tiled.cu/c3_cublas.cu exactly, so
    this is a genuine independent correctness check, not a repeat of the
    CUDA code. The grader compares the CUDA output element by element
    against this, which (unlike a single scalar checksum) rejects a kernel
    that gets the aggregate right while computing the wrong per-element
    attention."""
    import math

    def Q(i, t):
        return float(((i * 2 + t) % 11) - 5) / 2.0

    def K(j, t):
        return float(((j + t * 3) % 11) - 5) / 2.0

    def V(j, t):
        return float((j % 13) + 1) * (0.05 + 0.02 * t)

    scale = 1.0 / math.sqrt(float(d))
    out = []
    for i in range(N):
        s = [scale * sum(Q(i, t) * K(j, t) for t in range(d)) for j in range(N)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        denom = sum(e)
        p = [v / denom for v in e]
        for t in range(d):
            out.append(sum(p[j] * V(j, t) for j in range(N)))
    return out


def parse_out_line(stdout):
    """Pull the OUT,<v0>,<v1>,... element dump a Part-C binary prints at the
    small correctness size. Returns a list of floats, or None if absent."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("OUT,"):
            try:
                return [float(x) for x in line[4:].split(",")]
            except ValueError:
                return None
    return None


def max_abs_err(a, b):
    if a is None or b is None or not a or len(a) != len(b):
        return None
    # max() alone can overlook NaN depending on its position in the list.
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in zip(a, b)):
        return None
    return max(abs(x - y) for x, y in zip(a, b))


def build_and_run_part_c(mode, arch):
    partc_dir = os.path.join(SRC, "partc")
    result = {}
    correctness = {
        "c1_vs_c2_checksum_match": None,
        "c1_vs_c3_checksum_match": None,
        "reference_checksum_match": None,
    }

    if mode != "gpu":
        for name in ("c1_naive", "c2_tiled", "c3_cublas"):
            result[name] = program_result_skip("skipped: no GPU (dry-run mode)")
        result["correctness"] = correctness
        return result

    # Correctness sizes (N d), small enough for the element dump and chosen
    # to cross tile boundaries (129 and 257 are not multiples of the 32- or
    # 128-wide tiles), which catches masking and boundary bugs a single
    # aligned size would miss. The spec's full N=2048, d=64 defaults (no CLI
    # args) are used only for the timing run.
    sizes = [[8, 4], [129, 4], [257, 8]]
    TOL = 1e-6  # element-wise tolerance; fp64 attention agreement is far tighter than this

    ok_all, log_all = make_build(partc_dir, arch, targets=["c1_naive", "c2_tiled", "c3_cublas"])
    if not ok_all:
        reason = "make failed:\n" + log_all[-2000:]
        for name in ("c1_naive", "c2_tiled", "c3_cublas"):
            result[name] = program_result_fail(reason, build_ok=False)

    # outputs[name][size_idx] = flat list of N*d doubles from the OUT dump.
    outputs = {}
    for name in ("c1_naive", "c2_tiled", "c3_cublas"):
        if name in result:  # already marked failed above (build failed)
            continue
        bin_path = os.path.join(partc_dir, name)
        if not os.path.exists(bin_path):
            result[name] = program_result_fail("binary not found after build", build_ok=False)
            continue
        outs, rows_all = [], []
        for sz in sizes:
            out = run_binary_stdout(bin_path, sz, cwd=partc_dir, timeout=120)
            outs.append(parse_out_line(out))
            rows_all += parse_result_lines(out)
        rows_full = [coerce_run_row(r) for r in run_binary(bin_path, [[]], cwd=partc_dir, timeout=300)]
        rows = [coerce_run_row(r) for r in rows_all] + rows_full
        runs_ok = (all(o is not None and len(o) == N * d
                       and all(math.isfinite(v) for v in o)
                       for o, (N, d) in zip(outs, sizes))
                   and len(rows_full) == 1
                   and all(r.get("returncode") == 0 for r in rows_full))
        if runs_ok:
            result[name] = program_result_ok(rows)
            outputs[name] = outs
        else:
            result[name] = program_result_fail(
                "a correctness or timing run failed or produced incomplete output",
                build_ok=True, runs=rows)
            outputs[name] = [None] * len(sizes)

    # Independent reference, per size.
    refs = [reference_attention_output(N, d) for N, d in sizes]

    def all_sizes_match(a_list, b_list):
        """True iff every size's element dump is present and within TOL."""
        if a_list is None or b_list is None:
            return None
        oks = []
        for a, b in zip(a_list, b_list):
            e = max_abs_err(a, b)
            if e is None:
                return False
            oks.append(e < TOL)
        return all(oks) if oks else None

    if "c1_naive" in outputs:
        correctness["reference_checksum_match"] = all_sizes_match(outputs["c1_naive"], refs)
    if "c1_naive" in outputs and "c2_tiled" in outputs:
        correctness["c1_vs_c2_checksum_match"] = all_sizes_match(outputs["c2_tiled"], refs)
    if "c1_naive" in outputs and "c3_cublas" in outputs:
        correctness["c1_vs_c3_checksum_match"] = all_sizes_match(outputs["c3_cublas"], refs)

    result["correctness"] = correctness
    return result


def self_validate(results, schema_path):
    """Best-effort local schema check (jsonschema is not a hard dependency
    of this script -- Gradescope will validate results.json regardless)."""
    try:
        import jsonschema
    except ImportError:
        print("[grade_local.py] (jsonschema not installed -- skipping local schema check; "
              "Gradescope will still validate results.json)")
        return True
    try:
        with open(schema_path) as f:
            schema = json.load(f)
        jsonschema.validate(instance=results, schema=schema)
        print("[grade_local.py] results.json validated against results.schema.json: OK")
        return True
    except Exception as e:
        print(f"[grade_local.py] WARNING: results.json failed schema validation: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="Force the CPU-only dry run even if a GPU is detected.")
    ap.add_argument("--arch", default="sm_75",
                     help="nvcc target architecture (default sm_75, T4). "
                          "V100=sm_70, A100=sm_80, H100=sm_90, RTX 30xx=sm_86.")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"),
                     help="Where to write results.json (default: alongside this script).")
    args = ap.parse_args()

    host = detect_host()
    has_gpu = bool(host["cuda_available"] and which("nvcc"))
    mode = "dry-run" if (args.dry_run or not has_gpu) else "gpu"

    print(f"[grade_local.py] mode={mode} arch={args.arch} host={host['hostname']}")
    if mode == "dry-run":
        print("[grade_local.py] Running in CPU-only dry-run mode: every GPU-only check will")
        print("[grade_local.py] be marked 'skipped: no GPU'. This does NOT satisfy the")
        print("[grade_local.py] assignment -- rerun this on a real GPU (mode should say 'gpu')")
        print("[grade_local.py] before you submit.")

    results = {
        "meta": {
            "assignment": "hw3-cuda",
            "mode": mode,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "grader_version": "1.0",
            "host": host,
        },
        "part_a": build_and_run_part_a(mode, args.arch),
        "part_b": build_and_run_part_b(mode, args.arch),
        "part_c": build_and_run_part_c(mode, args.arch),
        "part_c_bonus_triton": {
            "status": "skipped",
            "reason": ("The Triton bonus (src/partc/c4_triton.ipynb) is graded by the TA on "
                       "a rubric, not by this script -- it needs a GPU notebook run, which "
                       "grade_local.py does not drive."),
            "note": None,
        },
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[grade_local.py] Wrote {args.out}")

    schema_path = os.path.join(HERE, "results.schema.json")
    if os.path.exists(schema_path):
        ok = self_validate(results, schema_path)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
