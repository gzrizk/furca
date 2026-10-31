# FURCA

A Python toolkit for the study of planar branching structures, developed on
the venation of Palaeozoic fern foliage.

FURCA represents branching structures as ordered, weighted trees and
measures their differences with the **Homology-Constrained Tree Distance
(HCTD)**, a top-down tree edit distance that accounts for positional
homology.

## Contents

- `furca/hctd.py` — HCTD implementation: per-tree size, pairwise distance,
  batch matrix, and the additive decomposition into metric and collapse
  components.
- `demo.py` — computes the HCTD matrix over the demonstration dataset.
- `scripts/pipeline/get_graphs.py` — extracts a tree from an annotated mask.
- `data/demo/graphml/` — ten real lateral vein modules, extracted from the
  author's annotations.

## Run

    pip install -e .
    python demo.py

Writes `results/hctd_matrix.csv` and prints the matrix. If `networkx` and
`pandas` are already installed, `python demo.py` also works without
installing the package.

## Extraction

`scripts/pipeline/get_graphs.py` extracts a tree from an annotated mask and
writes GraphML plus a quality-control overlay. The trees in `data/demo/` are
the output of this step. It requires a directory of annotated PNG masks and
a metadata table with the mm-per-pixel scale of each specimen; see
`config.yaml`.

    pip install -e ".[extract]"
    python scripts/pipeline/get_graphs.py

## Not included

Ordination, the discrete morphospace and the specimen images are not part of
this repository yet.

## Licence

Code and data are licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), see
`LICENSE`.

You may share the unmodified material for non-commercial purposes, with
attribution. Modifications, derivative works and commercial use require
written permission.

The licence will be revised when the accompanying manuscript is published.

Copyright (c) 2026 Gonzalo Rodriguez Rizk. See `NOTICE`.

## Author

Gonzalo Rodriguez Rizk
Museo Argentino de Ciencias Naturales "Bernardino Rivadavia", Buenos Aires, Argentina

## Citation

See `CITATION.cff`.
