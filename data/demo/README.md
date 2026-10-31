# Demonstration dataset

Ten lateral vein modules (LVMs) from Permian marattialean fern foliage,
Estancia La Golondrina Formation, Santa Cruz Province, Argentina. Specimens
LILPB1443, LILPB1447 and LILPB1453, housed at the Museo Argentino de
Ciencias Naturales "Bernardino Rivadavia" (LILPB collection).

Each file is a directed, ordered, weighted tree in GraphML, extracted from a
manual annotation of a specimen photograph.

- Nodes: `x`, `y` (pixel coordinates) and `node_role`.
- Edges: `weight` (pixels), `length_mm`, `curve_points`, and `child_index`
  on branching edges (1 = apical, 2 = basal).
- Graph: `filename`, `mm_per_pixel`, `edge_length_method`.

Reuse is subject to the same terms as the code: see `../../LICENSE`.
