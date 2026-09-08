"""Compare residual or operator-construction peak RSS between source trees.

This is a manual maintainer probe rather than a CI test. Each measurement runs
in a fresh process, loads the same saved CSR matrix before establishing its RSS
baseline, and measures either residual-representation construction or operator
statistics. In operator mode the representation is built before the baseline
so the measured increment isolates memory used to center the operator and
calculate its summary statistics. The parent polls current RSS while the worker
also reports its process-lifetime high-water mark as a cross-check.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import queue
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}

FAMILIES = (
    ("poisson", "pearson"),
    ("binomial", "pearson"),
    ("scaled_nb", "pearson"),
    ("poisson", "deviance"),
    ("binomial", "deviance"),
    ("scaled_nb", "deviance"),
)
DTYPES = ("float32", "float64")
PHASES = ("representation", "operator")


def _ru_maxrss_bytes() -> int:
    """Return the process-lifetime RSS high-water mark in bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _worker_main(arguments: list[str]) -> int:
    """Load one saved matrix and run one isolated measurement phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--dtype", choices=DTYPES, required=True)
    parser.add_argument("--clip", required=True)
    args = parser.parse_args(arguments)

    import numpy as np
    from scipy import sparse

    import sparse_count_pca
    from sparse_count_pca._operator import SparseLowRankLinearOperator
    from sparse_count_pca._residuals import build_residual_representation

    counts = sparse.load_npz(args.matrix).tocsr()
    row_totals = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    column_totals = np.asarray(counts.sum(axis=0, dtype=np.float64)).ravel()
    proportions = column_totals / column_totals.sum(dtype=np.float64)
    alpha = (
        np.full(counts.shape[1], 0.1, dtype=np.float64)
        if args.model == "scaled_nb"
        else None
    )
    clip = None if args.clip == "none" else float(args.clip)

    keywords = {
        "model": args.model,
        "residual": args.residual,
        "alpha": alpha,
        "clip": clip,
        "clip_mode": "symmetric",
        "clip_max_nnz_ratio": 1.0,
    }
    if "dtype" in inspect.signature(build_residual_representation).parameters:
        keywords["dtype"] = args.dtype

    def build_representation():
        return build_residual_representation(
            counts,
            row_totals,
            proportions,
            **keywords,
        )

    representation = build_representation() if args.phase == "operator" else None

    ready = {
        "event": "ready",
        "phase": args.phase,
        "pid": os.getpid(),
        "module_path": str(Path(sparse_count_pca.__file__).resolve()),
        "ru_maxrss_bytes": _ru_maxrss_bytes(),
    }
    print("MEMORY_READY " + json.dumps(ready), flush=True)
    if sys.stdin.readline().strip() != "go":
        raise RuntimeError("Memory worker did not receive the go signal")

    started = time.perf_counter()
    if args.phase == "representation":
        representation = build_representation()
        output_checksum = None
    else:
        assert representation is not None
        operator = SparseLowRankLinearOperator(
            representation,
            center=True,
            dtype=args.dtype,
            copy=False,
        )
        output_checksum = (
            operator.frobenius_squared_uncentered()
            + operator.frobenius_squared_centered()
        )
    elapsed = time.perf_counter() - started
    assert representation is not None
    done = {
        "event": "done",
        "phase": args.phase,
        "elapsed_seconds": elapsed,
        "ru_maxrss_bytes": _ru_maxrss_bytes(),
        "output_nnz": representation.sparse.nnz,
        "output_dtype": str(representation.sparse.dtype),
        "output_checksum": output_checksum,
    }
    print("MEMORY_DONE " + json.dumps(done), flush=True)
    return 0


def _source_directory(path: Path) -> Path:
    """Resolve a repository root or import root to its package source directory."""
    path = path.expanduser().resolve()
    source = path / "src" if (path / "src" / "sparse_count_pca").is_dir() else path
    if not (source / "sparse_count_pca").is_dir():
        raise ValueError(f"No sparse_count_pca package found under {path}")
    return source


def _current_rss_bytes(pid: int) -> int | None:
    """Read a live worker's resident set size with the system ps command."""
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    return int(output) * 1024 if result.returncode == 0 and output else None


def _parse_message(line: str, prefix: str) -> dict[str, Any]:
    """Parse one JSON worker-protocol message with the expected prefix."""
    if not line.startswith(prefix):
        raise RuntimeError(f"Expected {prefix.strip()}, received: {line.strip()}")
    return json.loads(line.removeprefix(prefix))


def _run_worker(
    *,
    label: str,
    source: Path,
    matrix: Path,
    phase: str,
    model: str,
    residual: str,
    dtype: str,
    clip: float | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Measure one isolated residual build after its input-load baseline."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        phase,
        "--matrix",
        str(matrix),
        "--model",
        model,
        "--residual",
        residual,
        "--dtype",
        dtype,
        "--clip",
        "none" if clip is None else str(clip),
    ]
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["PYTHONPATH"] = str(source)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=matrix.parent,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        stdout_lines = []
        ready_lines = queue.Queue()

        def drain_stdout() -> None:
            for line in process.stdout:
                stdout_lines.append(line)
                if line.startswith("MEMORY_READY "):
                    ready_lines.put(line)
            ready_lines.put(None)

        stdout_reader = threading.Thread(target=drain_stdout, daemon=True)
        stdout_reader.start()
        deadline = time.monotonic() + timeout_seconds

        def collect_output(*, kill: bool) -> tuple[str, str]:
            if kill and process.poll() is None:
                process.kill()
            process.wait()
            stdout_reader.join()
            stderr_file.flush()
            stderr_file.seek(0)
            return "".join(stdout_lines), stderr_file.read()

        try:
            ready_line = ready_lines.get(timeout=timeout_seconds)
        except queue.Empty:
            stdout, stderr = collect_output(kill=True)
            raise TimeoutError(
                f"{label} worker did not become ready within "
                f"{timeout_seconds:g} seconds:\n{stderr}\n{stdout}"
            ) from None
        if ready_line is None:
            stdout, stderr = collect_output(kill=False)
            raise RuntimeError(
                f"{label} worker exited before its baseline:\n{stderr}\n{stdout}"
            )
        ready = _parse_message(ready_line, "MEMORY_READY ")
        module_path = Path(ready["module_path"])
        if not module_path.is_relative_to(source):
            stdout, stderr = collect_output(kill=True)
            raise RuntimeError(
                f"{label} imported {module_path}, not code under {source}:\n"
                f"{stderr}\n{stdout}"
            )

        baseline = _current_rss_bytes(process.pid)
        if baseline is None:
            stdout, stderr = collect_output(kill=True)
            raise RuntimeError(
                f"Could not read baseline RSS for {label} worker:\n{stderr}\n{stdout}"
            )
        sampled_peak = baseline
        try:
            process.stdin.write("go\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            stdout, stderr = collect_output(kill=True)
            raise RuntimeError(
                f"{label} worker exited during its start handshake:\n{stderr}\n{stdout}"
            ) from error
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stdout, stderr = collect_output(kill=True)
                raise TimeoutError(
                    f"{label} {phase} {model}/{residual}/{dtype} worker exceeded "
                    f"{timeout_seconds:g} seconds:\n{stderr}\n{stdout}"
                )
            current = _current_rss_bytes(process.pid)
            if current is not None:
                sampled_peak = max(sampled_peak, current)
            time.sleep(min(poll_seconds, remaining))

        stdout, stderr = collect_output(kill=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"{label} {phase} {model}/{residual}/{dtype} worker failed:\n"
                f"{stderr}\n{stdout}"
            )
        done_lines = [
            line for line in stdout.splitlines() if line.startswith("MEMORY_DONE ")
        ]
        if len(done_lines) != 1:
            raise RuntimeError(f"Expected one completion message, received:\n{stdout}")
        done = _parse_message(done_lines[0], "MEMORY_DONE ")
        lifetime_peak = int(done["ru_maxrss_bytes"])
        ready_lifetime_peak = int(ready["ru_maxrss_bytes"])
    return {
        "source": label,
        "phase": phase,
        "model": model,
        "residual": residual,
        "clip": "none" if clip is None else "symmetric",
        "dtype": dtype,
        "elapsed_seconds": float(done["elapsed_seconds"]),
        "baseline_bytes": baseline,
        "sampled_peak_bytes": sampled_peak,
        "incremental_peak_bytes": max(0, sampled_peak - baseline),
        "lifetime_peak_bytes": lifetime_peak,
        "incremental_hwm_bytes": max(0, lifetime_peak - ready_lifetime_peak),
        "output_nnz": int(done["output_nnz"]),
        "output_dtype": done["output_dtype"],
    }


def _format_mib(value: float) -> str:
    """Format a byte count as mebibytes."""
    return f"{value / 2**20:.1f}"


def _print_run(record: dict[str, Any], repeat: int) -> None:
    """Print one compact measurement row."""
    print(
        f"{record['source']:9} r{repeat} {record['phase']:14} "
        f"{record['model']:9} "
        f"{record['residual']:8} {record['clip']:9} {record['dtype']:7} "
        f"stored={record['output_dtype']:7} "
        f"peak+={_format_mib(record['incremental_peak_bytes']):>7} MiB "
        f"hwm+={_format_mib(record['incremental_hwm_bytes']):>7} MiB "
        f"time={record['elapsed_seconds']:.3f}s"
    )


def _print_comparison(records: list[dict[str, Any]]) -> None:
    """Print median candidate-versus-baseline incremental peak RSS."""
    print(f"\nMedian sampled incremental peak RSS ({records[0]['phase']})")
    print("model     residual clip      dtype    baseline  candidate  reduction")
    keys = list(
        dict.fromkeys(
            (
                record["model"],
                record["residual"],
                record["clip"],
                record["dtype"],
            )
            for record in records
        )
    )
    for model, residual, clip, dtype in keys:
        selected = [
            record
            for record in records
            if (
                record["model"],
                record["residual"],
                record["clip"],
                record["dtype"],
            )
            == (model, residual, clip, dtype)
        ]
        values = {
            source: statistics.median(
                record["incremental_peak_bytes"]
                for record in selected
                if record["source"] == source
            )
            for source in ("baseline", "candidate")
        }
        reduction = (
            100.0 * (values["baseline"] - values["candidate"]) / values["baseline"]
            if values["baseline"]
            else math.nan
        )
        print(
            f"{model:9} {residual:8} {clip:9} {dtype:7} "
            f"{_format_mib(values['baseline']):>8} "
            f"{_format_mib(values['candidate']):>9} {reduction:8.1f}%"
        )


def _fixed_row_support_counts(
    n_obs: int,
    n_vars: int,
    nnz_per_row: int,
    rng: Any,
):
    """Build deterministic counts with exactly ``nnz_per_row`` entries per row."""
    import numpy as np
    from scipy import sparse

    nnz = n_obs * nnz_per_row
    indptr = np.arange(n_obs + 1, dtype=np.int64) * nnz_per_row
    indices = np.empty(nnz, dtype=np.int32)
    base_columns = np.arange(nnz_per_row, dtype=np.int64)
    rows_per_chunk = max(1, 1_000_000 // nnz_per_row)
    for row_start in range(0, n_obs, rows_per_chunk):
        row_stop = min(n_obs, row_start + rows_per_chunk)
        offsets = np.arange(row_start, row_stop, dtype=np.int64) * 104_729 % n_vars
        block = np.sort((offsets[:, None] + base_columns) % n_vars, axis=1)
        value_start = row_start * nnz_per_row
        value_stop = row_stop * nnz_per_row
        indices[value_start:value_stop] = block.ravel()
    data = rng.integers(1, 11, size=nnz, dtype=np.int32)
    counts = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_obs, n_vars),
    )
    assert counts.has_canonical_format, "generated counts must be canonical CSR"
    return counts


def _main(arguments: list[str]) -> int:
    """Generate one input and run every paired memory configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="representation")
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--n-obs", type=int, default=3000)
    parser.add_argument("--n-vars", type=int, default=3750)
    parser.add_argument("--density", type=float)
    parser.add_argument("--nnz-per-row", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--poll-ms", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--clip", type=float)
    parser.add_argument(
        "--clip-state",
        choices=("none", "symmetric", "both"),
        default="both",
    )
    parser.add_argument(
        "--model",
        choices=tuple(dict.fromkeys(model for model, _ in FAMILIES)),
        action="append",
        dest="models",
    )
    parser.add_argument(
        "--residual",
        choices=tuple(dict.fromkeys(residual for _, residual in FAMILIES)),
        action="append",
        dest="residuals",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        action="append",
        dest="dtypes",
    )
    args = parser.parse_args(arguments)
    if args.n_obs <= 0 or args.n_vars <= 0 or args.repeats <= 0:
        parser.error("matrix dimensions and repeats must be positive")
    if args.density is not None and args.nnz_per_row is not None:
        parser.error("density and nnz-per-row are mutually exclusive")
    if args.density is not None and not 0.0 < args.density <= 1.0:
        parser.error("density must be in (0, 1]")
    if args.nnz_per_row is not None and not 1 <= args.nnz_per_row <= args.n_vars:
        parser.error("nnz-per-row must be in [1, n_vars]")
    if args.poll_ms <= 0:
        parser.error("poll-ms must be positive")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.clip is not None and args.clip <= 0:
        parser.error("clip must be positive")

    baseline_source = _source_directory(args.baseline_source)
    candidate_source = _source_directory(args.candidate_source)
    clip_value = args.clip if args.clip is not None else math.sqrt(args.n_obs / 30)

    import numpy as np
    from scipy import sparse

    rng = np.random.default_rng(20260813)
    if args.nnz_per_row is None:
        density = 0.16 if args.density is None else args.density
        counts = sparse.random(
            args.n_obs,
            args.n_vars,
            density=density,
            format="csr",
            dtype=np.int32,
            random_state=rng,
            data_rvs=lambda size: rng.integers(1, 11, size=size, dtype=np.int32),
        )
    else:
        counts = _fixed_row_support_counts(
            args.n_obs,
            args.n_vars,
            args.nnz_per_row,
            rng,
        )
    if np.asarray(counts.sum(axis=1)).min() == 0:
        raise RuntimeError(
            "Generated matrix contains an empty row; choose more density"
        )

    records: list[dict[str, Any]] = []
    selected_models = set(args.models or (model for model, _ in FAMILIES))
    selected_residuals = set(args.residuals or (residual for _, residual in FAMILIES))
    selected_dtypes = args.dtypes or DTYPES
    if args.clip_state == "none":
        clip_values = (None,)
    elif args.clip_state == "symmetric":
        clip_values = (clip_value,)
    else:
        clip_values = (None, clip_value)
    configurations = [
        (model, residual, clip, dtype)
        for model, residual in FAMILIES
        if model in selected_models and residual in selected_residuals
        for clip in clip_values
        for dtype in selected_dtypes
    ]
    total_runs = args.repeats * len(configurations) * 2
    print(
        f"Input: {counts.shape[0]} x {counts.shape[1]}, nnz={counts.nnz:,}, "
        f"density={counts.nnz / math.prod(counts.shape):.4f}, "
        f"clip={clip_value:.6g}"
    )
    print(
        f"Runs: {total_runs}; polling every {args.poll_ms:g} ms; "
        f"worker timeout={args.timeout:g} s\n"
    )

    with tempfile.TemporaryDirectory(prefix="scp-memory-probe-") as temp_name:
        matrix = Path(temp_name) / "counts.npz"
        sparse.save_npz(matrix, counts, compressed=False)
        del counts
        for repeat in range(1, args.repeats + 1):
            for model, residual, clip, dtype in configurations:
                for label, source in (
                    ("baseline", baseline_source),
                    ("candidate", candidate_source),
                ):
                    record = _run_worker(
                        label=label,
                        source=source,
                        matrix=matrix,
                        phase=args.phase,
                        model=model,
                        residual=residual,
                        dtype=dtype,
                        clip=clip,
                        poll_seconds=args.poll_ms / 1000,
                        timeout_seconds=args.timeout,
                    )
                    records.append(record)
                    _print_run(record, repeat)

    _print_comparison(records)
    print(
        "\nThis probe reports evidence, not a machine-independent pass/fail "
        "threshold. Review clipped candidate peaks against both the baseline "
        "and the candidate's unclipped configuration."
    )
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_arguments = sys.argv[1:]
        worker_arguments.remove("--worker")
        raise SystemExit(_worker_main(worker_arguments))
    raise SystemExit(_main(sys.argv[1:]))
