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
- `data/demo/graphml/` — ten real lateral vein modules, extracted from the
  author's annotations.

## Run

    pip install -e .
    python demo.py

Writes `results/hctd_matrix.csv` and prints the matrix. If `networkx` and
`pandas` are already installed, `python demo.py` also works without
installing the package.

## Not included

Mask-to-tree extraction, ordination, visual quality control and the specimen
images are not part of this repository yet.

## Licence and reuse

Code and data are shared for transparency and review only. Reuse and
redistribution are not permitted until the accompanying manuscript is
published. See `LICENSE`.

## Author

Gonzalo Rodriguez Rizk
Museo Argentino de Ciencias Naturales "Bernardino Rivadavia", Buenos Aires, Argentina

## Citation

See `CITATION.cff`.
