"""
Public tests for HW3 (CUDA Kernels).

Run these yourself before submitting:
    pip install -r requirements.txt
    pytest tests/ -v

These do NOT require a GPU or nvcc. They check the things that are
checkable without one: that results.schema.json is a valid JSON Schema,
that grade_local.py's --dry-run mode runs cleanly and produces a
schema-valid results.json (with every GPU-only field marked "skipped: no
GPU"), that each required source file is present and non-empty, and that
every Makefile lists the binaries the assignment expects.

They do NOT check that your CUDA/cuBLAS kernels are correct or fast --
that requires a real GPU. Run `python grade_local.py` (no --dry-run) on
your own GPU machine for that; see the top-level README.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCHEMA_PATH = ROOT / "results.schema.json"
GRADE_SCRIPT = ROOT / "grade_local.py"


def test_schema_file_is_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft7Validator.check_schema(schema)


@pytest.fixture(scope="module")
def dry_run_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("hw3_dry_run") / "results.json"
    proc = subprocess.run(
        [sys.executable, str(GRADE_SCRIPT), "--dry-run", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"grade_local.py --dry-run exited nonzero.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert out.exists(), "grade_local.py --dry-run did not write results.json"
    return json.loads(out.read_text())


def test_dry_run_produces_schema_valid_results(dry_run_results):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=dry_run_results, schema=schema)


def test_dry_run_mode_is_dry_run_and_marks_gpu_parts_skipped(dry_run_results):
    assert dry_run_results["meta"]["mode"] == "dry-run"
    # Part A and Part C are entirely GPU-only; every program there must be
    # explicitly marked skipped (not silently omitted, not "ok").
    for prog in ("vecadd00", "vecadd01", "matmult00", "matmult01"):
        assert dry_run_results["part_a"][prog]["status"] == "skipped"
    for prog in ("c1_naive", "c2_tiled", "c3_cublas"):
        assert dry_run_results["part_c"][prog]["status"] == "skipped"
    # q2/q3 in Part B are also GPU-only.
    assert dry_run_results["part_b"]["q2_malloc"]["status"] == "skipped"
    assert dry_run_results["part_b"]["q3_unified"]["status"] == "skipped"


def test_dry_run_still_builds_and_runs_cpu_only_q1(dry_run_results):
    # q1_cpu needs only g++, not nvcc/a GPU -- the harness itself should be
    # exercised for real here, not blanket-skipped, so this test doubles as
    # a check that a completely-unmodified template still "runs" (and fails
    # correctness, since addArrays is a TODO) rather than crashing the
    # grader.
    q1 = dry_run_results["part_b"]["q1_cpu"]
    assert q1["status"] in ("ok", "failed"), q1
    assert q1["build_ok"] is True, (
        "q1_cpu should compile with plain g++ even with its TODO unfilled "
        f"(the stub body still compiles): {q1}"
    )


@pytest.mark.parametrize("makefile,expected_targets", [
    (SRC / "parta" / "Makefile", ["vecadd00", "vecadd01", "matmult00", "matmult01"]),
    (SRC / "partb" / "Makefile", ["q1_cpu", "q2_malloc", "q3_unified"]),
    (SRC / "partc" / "Makefile", ["c1_naive", "c2_tiled", "c3_cublas"]),
])
def test_makefile_lists_expected_binaries(makefile, expected_targets):
    text = makefile.read_text()
    for target in expected_targets:
        assert target in text, f"{makefile} does not mention target {target!r}"


@pytest.mark.parametrize("path", [
    SRC / "parta" / "vecaddKernel01.cu",
    SRC / "parta" / "matmultKernel01.cu",
    SRC / "partb" / "q1_cpu.cpp",
    SRC / "partb" / "q2_malloc.cu",
    SRC / "partb" / "q3_unified.cu",
    SRC / "partc" / "c1_naive.cu",
    SRC / "partc" / "c2_tiled.cu",
    SRC / "partc" / "c3_cublas.cu",
])
def test_required_files_present_and_nonempty(path):
    # The files you must submit should exist and have content. This catches
    # an accidentally deleted or blanked file, which is the real failure mode
    # the grader cannot recover from. It does NOT check whether you finished
    # the kernel (only a GPU run of grade_local.py does that), and removing
    # the TODO comments after you finish is fine.
    assert path.exists(), f"missing required file: {path}"
    assert path.read_text().strip(), f"required file is empty: {path}"
