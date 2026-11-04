#!/usr/bin/env python3
"""SwiftRel: correctness check against vcftools 0.1.17 + benchmark.

    python3 evaluate.py                    # build, random-input check, benchmark-set check + timing
    python3 evaluate.py --holdout          # the same on the five holdout sets
    python3 evaluate.py --quick            # build + random-input check only
    python3 evaluate.py --threads 1        # timing with SWIFTREL_THREADS=1

The check has two parts, and both must pass:
  * Fuzz: fresh random tiny VCFs every run (new seed each time, printed), random --relatedness2 options and
    site/sample filters, random SWIFTREL_THREADS. Each case runs the reference and the candidate in separate fresh directories with the same
    file names. When the reference succeeds, the candidate must exit 0 and produce byte-identical output (the
    .relatedness2 file, or stdout with --stdout). When the reference fails (non-zero exit or crash), the candidate must fail too.
    Log files and stderr are ignored. Cases where the reference itself times out are skipped and counted.
  * Benchmark sets: the candidate's .relatedness2 output must match the recorded sha256 of the reference output.
Timing: wall clock of the full command (parse + compute + write). The output file is deleted before each timed
repetition, outside the timed window, every repetition writes to a fresh name, and dirty pages are synced to
disk before the timer starts (without this, an A/A pair on identical binaries measured a 0.64 ratio).
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gen_data  # noqa: E402

MANIFEST = ROOT / "bench" / "manifest.json"
REF = ROOT / "ext" / "ref" / "vcftools"
CAND = ROOT / "solver" / "swiftrel"
FAIL_DIR = ROOT / "failures"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd, timeout, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env["SWIFTREL_THREADS"] = str(threads)
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", b"", b""


def ld_outputs(d):
    """Map of .relatedness2 file name -> bytes in directory d."""
    return {p.name: p.read_bytes() for p in sorted(Path(d).glob("*.relatedness2"))}


def build(solver_dir):
    binary = solver_dir / "swiftrel"
    binary.unlink(missing_ok=True)
    b = subprocess.run(["bash", str(solver_dir / "build.sh")], cwd=solver_dir.parent, capture_output=True, text=True,
                       timeout=900)
    if b.returncode != 0 or not binary.exists():
        print(b.stdout[-3000:], b.stderr[-3000:])
        return None
    return binary


def fuzz_gate(n_cases, seed, max_fail=5):
    rng = random.Random(seed)
    fails, skipped = 0, 0
    with tempfile.TemporaryDirectory() as td:
        for k in range(n_cases):
            data, suffix, args, extra = gen_data.fuzz_case(rng)
            fname = f"in_{rng.randrange(16**8):08x}{suffix}"
            threads = rng.choice([1, 2, 3, 4, 7])
            dirs = []
            for side in ("ref", "cand"):
                d = Path(td) / f"{side}_{k}"
                d.mkdir()
                (d / fname).write_bytes(data)
                for n, b in extra.items():
                    (d / n).write_bytes(b)
                dirs.append(d)
            inp = ["--gzvcf" if suffix.endswith(".gz") else "--vcf", fname]
            cmd_tail = [*inp, *args, "--out", "out"]
            rc_r, out_r, _ = run([str(REF), *cmd_tail], dirs[0], 20)
            if rc_r == "timeout":
                skipped += 1
                ok = True
            else:
                rc_c, out_c, err_c = run([str(CAND), *cmd_tail], dirs[1], 20, threads)
                if rc_r != 0:
                    ok = rc_c != 0 and rc_c != "timeout"
                elif "--stdout" in args:
                    ok = rc_c == 0 and out_c == out_r and ld_outputs(dirs[1]) == ld_outputs(dirs[0])
                else:
                    ok = rc_c == 0 and ld_outputs(dirs[1]) == ld_outputs(dirs[0])
            if not ok:
                fails += 1
                FAIL_DIR.mkdir(exist_ok=True)
                tag = f"fuzz_{seed}_{k}"
                (FAIL_DIR / (tag + suffix)).write_bytes(data)
                for n, b in extra.items():
                    (FAIL_DIR / (tag + "_" + n)).write_bytes(b)
                ref_files = {n: v[:600].decode(errors="replace") for n, v in ld_outputs(dirs[0]).items()}
                cand_files = {n: v[:600].decode(errors="replace") for n, v in ld_outputs(dirs[1]).items()}
                (FAIL_DIR / (tag + ".json")).write_text(json.dumps(dict(
                    input=tag + suffix, args=args, extra_files=list(extra), threads=threads,
                    cmd="<tool> " + " ".join(cmd_tail), ref_rc=rc_r, cand_rc=str(rc_c),
                    ref_stdout_head=out_r[:600].decode(errors="replace"),
                    cand_stdout_head=out_c[:600].decode(errors="replace"),
                    ref_out_heads=ref_files, cand_out_heads=cand_files,
                    cand_stderr_head=err_c[:600].decode(errors="replace")), indent=1))
                print(f"FUZZ FAIL case {k}: args={args} threads={threads} ref_rc={rc_r} cand_rc={rc_c} "
                      f"-> failures/{tag}.*")
                if fails >= max_fail:
                    break
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
    return fails, skipped


def timed_run(binary, row, td, threads, timeout):
    """One timed run of `binary` on a benchmark row. Returns (seconds or None, ok)."""
    src = ROOT / row["path"]
    link = Path(td) / (f"x{random.randrange(16**8):08x}" + "".join(src.suffixes))
    if not link.exists():
        os.symlink(src, link)
    prefix = f"o{random.randrange(16**8):08x}"
    out = Path(td) / (prefix + ".relatedness2")
    cmd = [str(binary), "--gzvcf" if link.name.endswith(".gz") else "--vcf", link.name, *row["args"], "--out", prefix]
    env = dict(os.environ, SWIFTREL_THREADS=str(threads))
    os.sync()   # flush the previous run's output so its writeback doesn't land inside this timed window
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, env=env)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    dt = time.perf_counter() - t0
    ok = (rc == 0 and out.exists() and out.stat().st_size == row["expected_bytes"]
          and sha256_file(out) == row["expected_sha256"])
    if not ok:
        size = out.stat().st_size if out.exists() else None
        print(f"BENCH FAIL {row['name']} ({binary.parent.parent.name}): rc={rc} bytes={size} "
              f"want={row['expected_bytes']}")
    for p in Path(td).glob(prefix + "*"):   # outside the timed window
        p.unlink()
    link.unlink()
    return dt, ok


def bench(rows, threads):
    results, fails = [], 0
    with tempfile.TemporaryDirectory() as td:
        for row in rows:
            timeout = max(60.0, 5.0 * row["ref_s"])
            times, ok = [], True
            for rep in range(3):
                dt, ok = timed_run(CAND, row, td, threads, timeout)
                if not ok:
                    break
                times.append(dt)
                if dt >= 5.0:
                    break
            if not ok:
                fails += 1
                results.append(dict(name=row["name"], real=row["real"], ok=False))
                continue
            best = min(times)
            results.append(dict(name=row["name"], real=row["real"], ok=True, seconds=round(best, 6),
                                speedup=round(row["ref_s"] / best, 3)))
    return results, fails


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0, help="SWIFTREL_THREADS for timing (default: all CPUs)")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    bad = [f for f, h in manifest["scripts_sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        print(f"SCRIPTS MODIFIED since prepare.py: {bad}. Restore them, or delete bench/ and rerun prepare.py.")
        sys.exit(2)

    t_start = time.time()
    if build(ROOT / "solver") is None:
        print("---\ngate:             BUILD_FAIL")
        sys.exit(1)

    threads = args.threads or manifest["threads"]
    seed = random.SystemRandom().randrange(10**9)
    fuzz_fail, fuzz_skip = fuzz_gate(args.fuzz, seed)
    mode = "quick" if args.quick else "holdout" if args.holdout else "dev"
    summary = dict(mode=mode, fuzz_seed=seed, fuzz_cases=args.fuzz, fuzz_failures=fuzz_fail,
                   fuzz_skipped_ref_timeouts=fuzz_skip, threads=threads)
    if not args.quick and fuzz_fail == 0:
        rows = manifest["holdout" if args.holdout else "dev"]
        res, bfail = bench(rows, threads)
        summary.update(bench_failures=bfail, per_dataset=res)
        if bfail == 0:
            summary.update(geomean_s=round(geomean([r["seconds"] for r in res]), 6),
                           speedup_vs_ref=round(geomean([r["speedup"] for r in res]), 3))
            real = [r["speedup"] for r in res if r["real"]]
            if real:
                summary["real_speedup_vs_ref"] = round(geomean(real), 3)
    elif not args.quick:
        summary["bench_failures"] = -1
    gate = fuzz_fail == 0 and (args.quick or summary.get("bench_failures") == 0)
    summary.update(gate="PASS" if gate else "FAIL", eval_seconds=round(time.time() - t_start, 1))
    name = {"holdout": "last_eval_holdout.json"}.get(mode, "last_eval.json")
    if args.threads and mode == "dev":
        name = f"last_eval_t{threads}.json"
    (ROOT / name).write_text(json.dumps(summary, indent=1))

    print("---")
    print(f"gate:             {summary['gate']}")
    print(f"mode:             {mode}")
    print(f"fuzz:             {args.fuzz} cases, seed {seed}, failures {fuzz_fail}, ref timeouts skipped {fuzz_skip}")
    if "geomean_s" in summary:
        print(f"geomean_s:        {summary['geomean_s']:.6f}")
        print(f"speedup_vs_ref:   {summary['speedup_vs_ref']:.3f}")
        if "real_speedup_vs_ref" in summary:
            print(f"real_speedup:     {summary['real_speedup_vs_ref']:.3f}")
        print("per_dataset:      " + "  ".join(f"{r['name']}={r['seconds']:.3f}s({r['speedup']:.1f}x)"
                                               for r in summary["per_dataset"]))
    print(f"threads:          {threads}")
    print(f"eval_seconds:     {summary['eval_seconds']}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
