#!/usr/bin/env python

from pathlib import Path
import argparse
import pickle
from collections import Counter


INDICATOR_MAP = {
    "Hashtags": "cohashtag",
    "URLs": "courl",
    "Retweet": "coretweet",
    "Co-token": "coword",
}


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_stats(file_path: Path):
    with file_path.open("rb") as handle:
        payload = pickle.load(handle)

    bipartite_edges = payload["bipartiteEdges"]
    if len(bipartite_edges) == 0:
        return {
            "n_users": 0,
            "n_items": 0,
            "i_over_n": 0.0,
            "n_over_i": 0.0,
            "users_per_item": 0.0,
            "avg_edge_weight": 0.0,
        }

    users = {user for user, _ in bipartite_edges}
    items = {item for _, item in bipartite_edges}

    edge_weight_counter = Counter(bipartite_edges)
    unique_pairs = list(edge_weight_counter.keys())

    n_users = len(users)
    n_items = len(items)

    users_per_item_counter = Counter(item for _, item in unique_pairs)

    total_edge_weight = len(bipartite_edges)
    n_unique_edges = len(unique_pairs)

    return {
        "n_users": n_users,
        "n_items": n_items,
        "i_over_n": safe_div(n_items, n_users),
        "n_over_i": safe_div(n_users, n_items),
        "users_per_item": safe_div(sum(users_per_item_counter.values()), n_items),
        "avg_edge_weight": safe_div(total_edge_weight, n_unique_edges),
    }


def format_row(label, stats):
    return (
        f"{label} & "
        f"{stats['n_users']:,} & "
        f"{stats['n_items']:,} & "
        f"{stats['i_over_n']:.4f} & "
        f"{stats['n_over_i']:.4f} & "
        f"{stats['users_per_item']:.2f} & "
        f"{stats['avg_edge_weight']:.2f} \\\\"
    )


def build_latex(rows):
    lines = [
        r"\begin{table}",
        r"    \small",
        r"    \centering",
        r"    \caption{Summary Statistics of bipartite graph. \#N=number of users, \#I=number of indicator}",
        r"    \label{tab:summary_stats}",
        r"    \begin{tabular}{lcccccc}",
        r"    \hline",
        r"    \textbf{Indicator} & ",
        r"    \textbf{\# N} & ",
        r"    \textbf{\# I} & ",
        r"    \textbf{I/N} & ",
        r"    \textbf{N/I} & ",
        r"    \textbf{Users per Indicator} & ",
        r"    \textbf{Avg. Edge Weight} \\",
        r"    \hline",
    ]

    for row in rows:
        lines.append(f"    {row}")

    lines.extend(
        [
            r"    \hline",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics table from bipartite edge pickle files."
    )
    parser.add_argument(
        "--dataset",
        default="hamas_israel_challenge_merged",
        help="Dataset name prefix used in <dataset>_<indicator>_bipartiteEdges.pkl",
    )
    parser.add_argument(
        "--networks-path",
        default="Outputs/Networks",
        help="Folder containing bipartite edge pickle files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .tex path. Defaults to Outputs/Tables/<dataset>_bipartite_summary_stats.tex",
    )

    args = parser.parse_args()

    networks_path = Path(args.networks_path)
    output_path = (
        Path(args.output)
        if args.output
        else Path("Outputs/Tables") / f"{args.dataset}_bipartite_summary_stats.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, indicator in INDICATOR_MAP.items():
        file_path = networks_path / f"{args.dataset}_{indicator}_bipartiteEdges.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        stats = compute_stats(file_path)
        rows.append(format_row(label, stats))

    latex = build_latex(rows)
    output_path.write_text(latex, encoding="utf-8")

    print(f"Saved LaTeX table to {output_path}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
