"""Compute the HCTD matrix over the demonstration LVMs."""

from pathlib import Path

import networkx as nx

from furca import hctd

DATA_DIR = Path(__file__).parent / "data" / "demo" / "graphml"
OUTPUT_CSV = Path(__file__).parent / "results" / "hctd_matrix.csv"


def main():
    paths = sorted(DATA_DIR.glob("*.graphml"))
    if not paths:
        raise SystemExit(f"No GraphML files found in {DATA_DIR}")

    graphs = {p.stem: nx.read_graphml(p) for p in paths}
    result = hctd(graphs)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.matrix.to_csv(OUTPUT_CSV)

    n = len(graphs)
    print(f"{n} LVMs, {n * (n - 1) // 2} pairwise comparisons")
    print(result.matrix.round(3))
    print(f"HCTD matrix saved at {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
