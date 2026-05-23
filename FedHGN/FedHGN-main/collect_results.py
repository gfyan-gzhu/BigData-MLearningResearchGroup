from pathlib import Path
import argparse
import math
import re

import numpy as np
import pandas as pd


KNOWN_EXP_FOLDERS = {"normal", "chaocan", "xiaorong"}
KNOWN_ABLATIONS = {"no_mp", "uniform", "static", "no_residual", "B", "C", "B+C"}


def parse_framework_dir(framework_dir: str):
    parts = framework_dir.split("_")
    if len(parts) >= 2 and parts[-1] in KNOWN_ABLATIONS:
        framework = "_".join(parts[:-1])
        ablation = parts[-1]
        method = ablation
    else:
        framework = framework_dir
        ablation = None
        method = "full"
    return framework, ablation, method


def parse_exp_name(exp_name: str):
    """
    expected style:
    AIFB_random-etypes_5_seed1000_mpb8_mpp16
    """
    tokens = exp_name.split("_")
    info = {
        "dataset": None,
        "split_strategy": None,
        "num_clients": None,
        "random_seed": None,
        "mp_num_gating_bases": None,
        "mp_max_paths": None,
    }

    if len(tokens) >= 1:
        info["dataset"] = tokens[0]
    if len(tokens) >= 2:
        info["split_strategy"] = tokens[1]
    if len(tokens) >= 3:
        try:
            info["num_clients"] = int(tokens[2])
        except Exception:
            info["num_clients"] = None

    for tok in tokens[3:]:
        if tok.startswith("seed"):
            try:
                info["random_seed"] = int(tok.replace("seed", ""))
            except Exception:
                pass
        elif tok.startswith("mpb"):
            try:
                info["mp_num_gating_bases"] = int(tok.replace("mpb", ""))
            except Exception:
                pass
        elif tok.startswith("mpp"):
            try:
                info["mp_max_paths"] = int(tok.replace("mpp", ""))
            except Exception:
                pass

    return info


def read_results_txt(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].strip().split("\t")
    values = lines[1].strip().split("\t")
    out = {}
    for h, v in zip(headers, values):
        try:
            out[h] = float(v)
        except Exception:
            out[h] = v
    return out


def scan_results(saves_root: Path):
    rows = []

    for res_file in saves_root.rglob("results.txt"):
        rel = res_file.relative_to(saves_root)
        parts = list(rel.parts)

        # new layout:
        # chaocan / FedSP-MPG_no_mp / RGCN / exp_name / 1 / results.txt
        if len(parts) >= 6 and parts[0] in KNOWN_EXP_FOLDERS:
            exp_folder = parts[0]
            framework_dir = parts[1]
            model = parts[2]
            exp_name = parts[3]
            run_id = parts[4]
        # old layout:
        # FedSP-MPG_no_mp / RGCN / exp_name / 1 / results.txt
        elif len(parts) >= 5:
            exp_folder = "normal"
            framework_dir = parts[0]
            model = parts[1]
            exp_name = parts[2]
            run_id = parts[3]
        else:
            continue

        framework, ablation, method = parse_framework_dir(framework_dir)
        info = parse_exp_name(exp_name)
        metrics = read_results_txt(res_file)

        row = {
            "exp_folder": exp_folder,
            "framework_dir": framework_dir,
            "framework": framework,
            "ablation": ablation,
            "method": method,
            "model": model,
            "exp_name": exp_name,
            "run_id": run_id,
            "result_path": str(res_file),
        }
        row.update(info)
        row.update(metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # numeric cast
    for col in df.columns:
        if col in {
            "exp_folder", "framework_dir", "framework", "ablation", "method",
            "model", "exp_name", "run_id", "result_path", "dataset", "split_strategy"
        }:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def get_metric_columns(df: pd.DataFrame):
    meta_cols = {
        "exp_folder", "framework_dir", "framework", "ablation", "method",
        "model", "exp_name", "run_id", "result_path",
        "dataset", "split_strategy", "num_clients", "random_seed",
        "mp_num_gating_bases", "mp_max_paths"
    }
    metric_cols = []
    for col in df.columns:
        if col in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)
    return metric_cols


def format_mean_std(mean_val, std_val):
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        std_val = 0.0
    return f"{mean_val:.4f} ± {std_val:.4f}"


def summarize_mean_std(df: pd.DataFrame, group_cols, metric_cols):
    if df.empty:
        return pd.DataFrame()

    out_rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {k: v for k, v in zip(group_cols, keys)}
        row["n_runs"] = len(g)

        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().astype(float).values
            if len(vals) == 0:
                mean_v = np.nan
                std_v = np.nan
            elif len(vals) == 1:
                mean_v = float(vals[0])
                std_v = 0.0
            else:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1))

            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
            row[f"{m}_mean±std"] = format_mean_std(mean_v, std_v)

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    return out_df.sort_values(group_cols).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saves-root", type=str, default="./saves")
    parser.add_argument("--out-dir", type=str, default="./summary")
    parser.add_argument("--fixed-gating-bases", type=int, default=8)
    parser.add_argument("--fixed-max-paths", type=int, default=16)
    args = parser.parse_args()

    saves_root = Path(args.saves_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_results(saves_root)
    if df.empty:
        print(f"[WARN] No results.txt found under: {saves_root}")
        return

    # save raw table
    df.to_csv(out_dir / "all_results_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_dir / 'all_results_raw.csv'}")

    metric_cols = get_metric_columns(df)

    # -------------------------
    # Hyperparameter summaries
    # -------------------------
    chaocan = df[df["exp_folder"] == "chaocan"].copy()

    if not chaocan.empty:
        # mp_num_gating_bases sweep:
        # keep rows with fixed mp_max_paths
        if "mp_max_paths" in chaocan.columns:
            mpb_df = chaocan[chaocan["mp_max_paths"] == args.fixed_max_paths].copy()
        else:
            mpb_df = pd.DataFrame()

        if not mpb_df.empty and "mp_num_gating_bases" in mpb_df.columns:
            mpb_summary = summarize_mean_std(
                mpb_df,
                group_cols=[
                    "dataset", "split_strategy", "num_clients",
                    "mp_num_gating_bases", "mp_max_paths"
                ],
                metric_cols=metric_cols,
            )
            mpb_summary.to_csv(
                out_dir / "chaocan_mpb_summary.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[OK] saved: {out_dir / 'chaocan_mpb_summary.csv'}")
        else:
            print("[WARN] cannot build chaocan_mpb_summary.csv")

        # mp_max_paths sweep:
        # keep rows with fixed mp_num_gating_bases
        if "mp_num_gating_bases" in chaocan.columns:
            mpp_df = chaocan[chaocan["mp_num_gating_bases"] == args.fixed_gating_bases].copy()
        else:
            mpp_df = pd.DataFrame()

        if not mpp_df.empty and "mp_max_paths" in mpp_df.columns:
            mpp_summary = summarize_mean_std(
                mpp_df,
                group_cols=[
                    "dataset", "split_strategy", "num_clients",
                    "mp_num_gating_bases", "mp_max_paths"
                ],
                metric_cols=metric_cols,
            )
            mpp_summary.to_csv(
                out_dir / "chaocan_mpp_summary.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[OK] saved: {out_dir / 'chaocan_mpp_summary.csv'}")
        else:
            print("[WARN] cannot build chaocan_mpp_summary.csv")
    else:
        print("[INFO] no chaocan results found")

    # -------------------------
    # Ablation tables
    # -------------------------
    xiaorong = df[df["exp_folder"] == "xiaorong"].copy()

    if not xiaorong.empty:
        # for your current setting, each ablation usually has only 1 seed
        xiaorong = xiaorong.sort_values(
            ["dataset", "split_strategy", "num_clients", "method", "random_seed"]
        ).reset_index(drop=True)

        xiaorong.to_csv(
            out_dir / "xiaorong_long.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[OK] saved: {out_dir / 'xiaorong_long.csv'}")

        for metric in metric_cols:
            table = xiaorong.pivot_table(
                index=["dataset", "split_strategy", "num_clients"],
                columns="method",
                values=metric,
                aggfunc="first",
            ).reset_index()

            # reorder common ablation columns
            common_order = ["full", "no_mp", "uniform", "static", "no_residual"]
            ordered_cols = ["dataset", "split_strategy", "num_clients"]
            for c in common_order:
                if c in table.columns:
                    ordered_cols.append(c)
            for c in table.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)

            table = table[ordered_cols]
            table.to_csv(
                out_dir / f"xiaorong_{metric}_table.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[OK] saved: {out_dir / f'xiaorong_{metric}_table.csv'}")
    else:
        print("[INFO] no xiaorong results found")


if __name__ == "__main__":
    main()