#!/usr/bin/env python3
import argparse
import csv
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_cmd(cmd):
    start = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print("[ERROR] Command failed:", " ".join(cmd), file=sys.stderr)
        print("[ERROR] STDOUT:\n", result.stdout, file=sys.stderr)
        print("[ERROR] STDERR:\n", result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return elapsed


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "repeat_idx", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary, path, title):
    x = [r["batch_size"] for r in summary]
    y = [r["mean_runtime_sec"] for r in summary]
    yerr = [r["std_runtime_sec"] for r in summary]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xlabel("Batch size")
    plt.ylabel("Runtime (hours)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def benchmark_feature(args, root_out):
    rows = []
    summary = []

    for bs in args.feat_batch_sizes:
        runtimes = []
        print(f"\n[FEATURE] batch_size={bs}")

        if args.warmup:
            warmup_out = root_out / f"feat_warmup_bs{bs}"
            clear_dir(warmup_out)
            cmd = [
                "conda", "run", "-n", args.reid_env,
                "python3", args.reid_script,
                "--tracklets_folder", args.tracklets_folder,
                "--output_folder", str(warmup_out),
                "--batch_size", str(bs),
            ]
            _ = run_cmd(cmd)

        for rep in range(args.repeats):
            feat_out = root_out / f"features_bs{bs}_rep{rep+1}"
            clear_dir(feat_out)

            cmd = [
                "conda", "run", "-n", args.reid_env,
                "python3", args.reid_script,
                "--tracklets_folder", args.tracklets_folder,
                "--output_folder", str(feat_out),
                "--batch_size", str(bs),
            ]
            elapsed = run_cmd(cmd)
            runtimes.append(elapsed)
            rows.append({
                "batch_size": bs,
                "repeat_idx": rep + 1,
                "runtime_sec": elapsed,
            })
            print(f"[FEATURE] rep={rep+1}, time={elapsed:.4f}s")

        summary.append({
            "batch_size": bs,
            "mean_runtime_sec": statistics.mean(runtimes),
            "std_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        })

    save_csv(rows, root_out / "feature_batch_benchmark.csv")
    save_plot(summary, root_out / "feature_runtime_vs_batch.png", "Feature generation runtime vs batch size")


def benchmark_legibility(args, root_out):
    rows = []
    summary = []

    for bs in args.leg_batch_sizes:
        runtimes = []
        print(f"\n[LEGIBILITY] batch_size={bs}")

        if args.warmup:
            warmup_result = root_out / f"legibility_warmup_bs{bs}.json"
            cmd = [
                "conda", "run", "-n", args.leg_env,
                "python3", args.leg_script,
                "--image_dir", args.leg_image_dir,
                "--model", args.leg_model,
                "--batch_size", str(bs),
                "--output", str(warmup_result),
            ]
            _ = run_cmd(cmd)

        for rep in range(args.repeats):
            result_file = root_out / f"legibility_result_bs{bs}_rep{rep+1}.json"
            cmd = [
                "conda", "run", "-n", args.leg_env,
                "python3", args.leg_script,
                "--image_dir", args.leg_image_dir,
                "--model", args.leg_model,
                "--batch_size", str(bs),
                "--output", str(result_file),
            ]
            elapsed = run_cmd(cmd)
            runtimes.append(elapsed)
            rows.append({
                "batch_size": bs,
                "repeat_idx": rep + 1,
                "runtime_sec": elapsed,
            })
            print(f"[LEGIBILITY] rep={rep+1}, time={elapsed:.4f}s")

        summary.append({
            "batch_size": bs,
            "mean_runtime_sec": statistics.mean(runtimes),
            "std_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        })

    save_csv(rows, root_out / "legibility_batch_benchmark.csv")
    save_plot(summary, root_out / "legibility_runtime_vs_batch.png", "Legibility runtime vs batch size")


def benchmark_str(args, root_out):
    rows = []
    summary = []

    for bs in args.str_batch_sizes:
        runtimes = []
        print(f"\n[STR] batch_size={bs}")

        if args.warmup:
            warmup_result = root_out / f"str_warmup_bs{bs}.json"
            cmd = [
                "conda", "run", "-n", args.str_env,
                "python3", args.str_script,
                args.str_model,
                f"--data_root={args.str_data_root}",
                f"--batch_size={bs}",
                "--inference",
                "--result_file", str(warmup_result),
            ]
            _ = run_cmd(cmd)

        for rep in range(args.repeats):
            result_file = root_out / f"str_result_bs{bs}_rep{rep+1}.json"
            cmd = [
                "conda", "run", "-n", args.str_env,
                "python3", args.str_script,
                args.str_model,
                f"--data_root={args.str_data_root}",
                f"--batch_size={bs}",
                "--inference",
                "--result_file", str(result_file),
            ]
            elapsed = run_cmd(cmd)
            runtimes.append(elapsed)
            rows.append({
                "batch_size": bs,
                "repeat_idx": rep + 1,
                "runtime_sec": elapsed,
            })
            print(f"[STR] rep={rep+1}, time={elapsed:.4f}s")

        summary.append({
            "batch_size": bs,
            "mean_runtime_sec": statistics.mean(runtimes),
            "std_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
        })

    save_csv(rows, root_out / "str_batch_benchmark.csv")
    save_plot(summary, root_out / "str_runtime_vs_batch.png", "STR runtime vs batch size")


def validate_args(args):
    if not (args.run_feature or args.run_legibility or args.run_str):
        raise ValueError("At least one stage must be enabled: --run_feature, --run_legibility, or --run_str")

    if args.run_feature:
        required = {
            "reid_script": args.reid_script,
            "tracklets_folder": args.tracklets_folder,
            "reid_env": args.reid_env,
            "feat_batch_sizes": args.feat_batch_sizes,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Feature stage enabled but missing arguments: {missing}")

    if args.run_legibility:
        required = {
            "leg_script": args.leg_script,
            "leg_image_dir": args.leg_image_dir,
            "leg_model": args.leg_model,
            "leg_env": args.leg_env,
            "leg_batch_sizes": args.leg_batch_sizes,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Legibility stage enabled but missing arguments: {missing}")

    if args.run_str:
        required = {
            "str_script": args.str_script,
            "str_model": args.str_model,
            "str_data_root": args.str_data_root,
            "str_env": args.str_env,
            "str_batch_sizes": args.str_batch_sizes,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"STR stage enabled but missing arguments: {missing}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_feature", action="store_true")
    parser.add_argument("--run_legibility", action="store_true")
    parser.add_argument("--run_str", action="store_true")

    parser.add_argument("--reid_script", type=str)
    parser.add_argument("--tracklets_folder", type=str)
    parser.add_argument("--reid_env", type=str)

    parser.add_argument("--leg_script", type=str)
    parser.add_argument("--leg_image_dir", type=str)
    parser.add_argument("--leg_model", type=str)
    parser.add_argument("--leg_env", type=str)

    parser.add_argument("--str_script", type=str)
    parser.add_argument("--str_model", type=str)
    parser.add_argument("--str_data_root", type=str)
    parser.add_argument("--str_env", type=str)

    parser.add_argument("--feat_batch_sizes", type=int, nargs="+")
    parser.add_argument("--leg_batch_sizes", type=int, nargs="+")
    parser.add_argument("--str_batch_sizes", type=int, nargs="+")

    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    validate_args(args)

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    if args.run_feature:
        benchmark_feature(args, root_out)

    if args.run_legibility:
        benchmark_legibility(args, root_out)

    if args.run_str:
        benchmark_str(args, root_out)

    print(f"\n[INFO] All outputs saved under: {root_out}")


if __name__ == "__main__":
    main()