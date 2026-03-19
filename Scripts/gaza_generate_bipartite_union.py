#!/usr/bin/env python

from pathlib import Path
import argparse
import pickle

from tqdm.auto import tqdm
import coordinationz as cz
import coordinationz.indicator_utilities as czind
import coordinationz.preprocess_utilities as czpre


DEFAULT_DATASET = "hamas_israel_challenge_merged"
DEFAULT_CONFIG = "config_union.toml"
DEFAULT_INDICATORS = ["cohashtag", "coretweet", "courl", "coword"]


def external_bipartite(file_path: Path):
    edges = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split(" ")
            user = tokens[1]
            multiplicity = int(float(tokens[2]))
            for _ in range(multiplicity):
                edges.append((user, tokens[0]))
    return edges


def text_similarity_partial(df, config, data_name):
    parameters = config["indicator"].get("textsimilarity", {})
    return czind.obtainBipartiteEdgesTextSimilarity(df, data_name, **parameters)


def main():
    tqdm.pandas()

    parser = argparse.ArgumentParser(
        description="Generate and save bipartite edges for the Gaza merged dataset using a union config."
    )
    parser.add_argument("dataname", nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("-i", "--indicators", nargs="+", default=DEFAULT_INDICATORS)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix inserted before '<dataname>_<indicator>_bipartiteEdges.pkl'.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = cz.load_config(str(config_path))
    print(f"Loading config from {config_path}...")

    networks_path = Path(config["paths"]["NETWORKS"]).resolve()
    networks_path.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed data...")
    df = czpre.loadPreprocessedData(args.dataname, config=config)

    run_parameters = czind.parseParameters(config, args.indicators)

    bipartite_method = {
        "coretweet": czind.obtainBipartiteEdgesRetweets,
        "cohashtag": czind.obtainBipartiteEdgesHashtags,
        "courl": czind.obtainBipartiteEdgesURLs,
        "coretweetusers": czind.obtainBipartiteEdgesRetweetsUsers,
        "coword": czind.obtainBipartiteEdgesWords,
        "textsimilarity": lambda frame: text_similarity_partial(frame, config, args.dataname),
    }

    for indicator in args.indicators:
        print(f"Generating bipartite edges for '{indicator}'...")
        df_filtered = czind.filterUsersByMinActivities(
            df,
            activityType=indicator,
            **run_parameters["user"][indicator],
        )

        if indicator == "usctextsimilarity":
            raise ValueError(
                "'usctextsimilarity' creates a direct user-user network, not a bipartite edge list."
            )

        if indicator.startswith("external"):
            file_path = Path(indicator.split(":", maxsplit=1)[1])
            allowed_users = set(df_filtered.user_id)
            bipartite_edges = [
                (user, item)
                for user, item in external_bipartite(file_path)
                if user in allowed_users
            ]
        else:
            if indicator not in bipartite_method:
                raise KeyError(f"Unsupported indicator: {indicator}")
            bipartite_edges = bipartite_method[indicator](df_filtered)

        bipartite_edges = czind.filterNodes(
            bipartite_edges,
            **run_parameters["filter"][indicator],
        )

        if not bipartite_edges:
            print(f"WARNING: no edges for '{indicator}' after filtering. Skipping save.")
            continue

        prefix = f"{args.output_prefix}_" if args.output_prefix else ""
        output_path = networks_path / f"{prefix}{args.dataname}_{indicator}_bipartiteEdges.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(
                {"bipartiteEdges": bipartite_edges, "dfFiltered": df_filtered},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"Saved {len(bipartite_edges):,} edges to {output_path}")


if __name__ == "__main__":
    main()
