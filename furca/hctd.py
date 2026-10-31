import math
from dataclasses import dataclass

import networkx as nx
import pandas as pd


# --- DATA CONTAINER ---


@dataclass(frozen=True)
class HCTDPairResult:
    """Decomposed HCTD between two LVMs.

    The full HCTD between two trees is additively decomposed into two
    components computed in the same traversal:

    - ``metric``   captures proportional reallocation within the
      homologous backbone (absolute differences in size-normalized
      branch lengths at paired internal edges whose subtrees are
      completely isomorphic).
    - ``collapse`` captures the fractional weight of structurally
      divergent subtrees (rupture cost: total normalized vein length
      below the first out-degree mismatch on each side, including
      the length difference of the parent edge leading into the
      rupture).

    Both components are raw (undivided) costs and satisfy
    ``total == metric + collapse`` exactly. Each lies in
    [0, Λ(T1) + Λ(T2)] and so does ``total``; there is no [0, 1]
    bound (see the B0 spec, DECISIONS.md).

    Attributes
    ----------
    metric : float
        Deformation-only component in [0, Λ1 + Λ2]; zero unless the
        paired subtrees are structurally isomorphic (same out-degree
        at every paired node).
    collapse : float
        Homology-rupture component in [0, Λ1 + Λ2].
    total : float
        Full HCTD; ``metric + collapse`` (raw cost ``C``).
    """
    metric: float
    collapse: float
    total: float


@dataclass(frozen=True)
class HCTDResult:
    """Output of batch HCTD computation.

    Mirrors the geomorph `gpagen()` return convention: the geometric
    content (size-normalized pairwise distances) and the size scalars
    are reported as complementary objects from a single call. The
    distance content is reported in three matrices: the full HCTD and
    its additive components (see ``HCTDPairResult``).

    Attributes
    ----------
    matrix : pandas.DataFrame
        n x n symmetric pairwise HCTD distance matrix. Values are raw
        costs (no per-pair divisor), i.e. bounded in [0, Λ_i + Λ_j]
        rather than [0, 1]. Index and columns are LVM identifiers
        (graphml stems).
    matrix_metric : pandas.DataFrame
        Metric (deformation) component matrix. Same shape, index, and
        bounds as ``matrix``.
    matrix_collapse : pandas.DataFrame
        Collapse (homology-rupture) component matrix. Same shape,
        index, and bounds as ``matrix``. Invariant:
        ``matrix == matrix_metric + matrix_collapse`` element-wise.
    sizes : pandas.Series
        Per-LVM size in mm: cumulative vein length along the longest
        root-to-tip path. Index is LVM identifiers; ``name == 'size_mm'``.
    """
    matrix: pd.DataFrame
    matrix_metric: pd.DataFrame
    matrix_collapse: pd.DataFrame
    sizes: pd.Series


@dataclass(frozen=True)
class HCTDTraceEvent:
    """One cost-accumulating step of the size-normalized HCTD traversal.

    Emitted, in traversal order, into the optional ``trace`` list passed
    to :func:`hctd_pair`. An event is produced wherever a cost is added;
    a paired node whose edge contributes exactly zero still emits an
    event (``amount == 0.0``), so the ledger is complete and its sums
    reconstruct both raw components exactly.

    Attributes
    ----------
    kind : str
        ``"metric"`` (paired parent edge inside the homologous
        backbone), ``"rupture_parent"`` (paired parent edge leading into
        a homology rupture), or ``"subtree"`` (normalized descendant
        vein length absorbed at a rupture).
    node1, node2 : object
        Paired nodes whose parent edges are compared. For a
        ``"subtree"`` event these are the rupture nodes themselves.
    parent1, parent2 : object or None
        Parents of ``node1`` / ``node2``; ``None`` at the roots.
    out_degree1, out_degree2 : int
        Out-degrees of ``node1`` / ``node2``.
    w1, w2 : float
        Size-normalized parent-edge lengths (``length_mm / size``).
        Zero for ``"subtree"`` events.
    amount : float
        Raw (undivided) cost contributed by this step; non-negative.
        ``sum(amount)`` over a component reproduces its raw value.
    component : str
        ``"metric"`` or ``"collapse"``.
    side : str
        ``"both"`` for paired-edge events, ``"1"`` or ``"2"`` for
        ``"subtree"`` events (which tree's divergent subtree).
    """
    kind: str
    node1: object
    node2: object
    parent1: object
    parent2: object
    out_degree1: int
    out_degree2: int
    w1: float
    w2: float
    amount: float
    component: str
    side: str


# --- TOPOLOGICAL HELPERS ---


def find_root(G):
    """Return the unique root of a DiGraph (node with in_degree == 0).

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph.

    Returns
    -------
    int or None
        Root node ID, or None if zero or more than one node has
        in_degree == 0 (caller decides how to handle).
    """
    roots = [n for n, d in G.in_degree() if d == 0]
    if len(roots) == 1:
        return roots[0]
    return None


def ordered_children(G, n):
    """Return the children of ``n`` ordered by ``child_index``.

    The graph must carry a ``child_index`` (1 = apical, ..., n = basal)
    on every edge leaving a branching node, as produced by the extraction
    pipeline (``assign_child_index``). Nodes with a single child have no
    rank, so the only child is returned as-is; childless nodes return an
    empty list.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed venation graph with ``child_index`` edge attributes.
    n : node
        Parent node.

    Returns
    -------
    list
        Children of ``n``, ordered by ``child_index`` ascending (apical
        to basal).

    Raises
    ------
    ValueError
        If a branching node (two or more children) has an edge missing
        ``child_index``.
    """
    children = list(G.successors(n))
    if len(children) < 2:
        return children
    try:
        return sorted(children, key=lambda c: G[n][c]["child_index"])
    except KeyError as exc:
        raise ValueError(
            f"ordered_children: node {n!r} has {len(children)} children "
            f"but edge {n!r}->{exc.args[0]!r} lacks 'child_index'. The "
            f"graph does not follow the extraction contract."
        ) from None


# --- LVM SIZE ---


def lvm_size(G):
    """Compute the size of an LVM as the longest root-to-tip vein length.

    The size is defined as
        size(G) = max over tips t of sum(length_mm along root->t path).

    This scalar is invariant to internal refinements (splitting an edge
    into a bifurcation with two equal-length daughters does not change
    the longest path) and grows only when the LVM extends further from
    the midvein insertion.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed venation graph with ``length_mm`` on every edge and a
        unique root (in_degree == 0).

    Returns
    -------
    float
        LVM size in millimetres.

    Raises
    ------
    ValueError
        If the graph has no unique root, has zero edges, or any edge
        lacks ``length_mm``.
    """
    root = find_root(G)
    if root is None:
        raise ValueError(
            "lvm_size: graph has no unique root (in_degree == 0). "
            "Cannot define a root-to-tip path."
        )

    if G.number_of_edges() == 0:
        raise ValueError("lvm_size: graph has no edges.")

    # BFS/DFS from root, accumulating max path length per node.
    cum_length = {root: 0.0}
    stack = [root]
    while stack:
        n = stack.pop()
        for child in G.successors(n):
            edge_data = G.get_edge_data(n, child)
            if "length_mm" not in edge_data:
                raise ValueError(
                    f"lvm_size: edge ({n}->{child}) lacks 'length_mm'."
                )
            cum_length[child] = cum_length[n] + edge_data["length_mm"]
            stack.append(child)

    return max(cum_length.values())


def d_size(G1, G2):
    """Log-ratio size distance between two LVMs.

    Defined as ``|log(lvm_size(G1) / lvm_size(G2))|``. Symmetric and
    additive on the multiplicative scale: if G2 = k·G1 then
    ``d_size(G1, G2) = |log k|``. Analogous to the log centroid-size
    difference used alongside Procrustes shape distances in geomorph.

    Parameters
    ----------
    G1, G2 : networkx.DiGraph
        Directed venation graphs with ``length_mm`` on every edge and a
        unique root.

    Returns
    -------
    float
        Non-negative size distance in log-mm units. Zero if and only if
        both LVMs have the same max root-to-tip path length.

    Raises
    ------
    ValueError
        If either graph has no unique root or non-positive size (via
        ``lvm_size``).
    """
    s1 = lvm_size(G1)
    s2 = lvm_size(G2)
    return abs(math.log(s1 / s2))


def _descendant_mass(G, root, size_mm):
    """Sum of (length_mm / size_mm) over edges descending from `root`.

    Excludes any edge reaching `root` (only edges strictly below).
    Used at homology ruptures to penalize the divergent subtree.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed venation graph with ``length_mm`` on edges.
    root : int
        Node from which descendant edges are summed.
    size_mm : float
        LVM size used to normalize edge lengths. Should equal
        ``lvm_size(G)``; passed in to avoid recomputation.

    Returns
    -------
    float
        Sum of normalized descendant edge lengths (dimensionless).
    """
    total = 0.0
    for child in G.successors(root):
        edge_data = G.get_edge_data(root, child)
        if edge_data and "length_mm" in edge_data:
            total += edge_data["length_mm"] / size_mm
        total += _descendant_mass(G, child, size_mm)
    return total


# --- HCTD ---


def hctd_pair(G1, G2, trace=None):
    """Compute HCTD between two venation graphs, decomposed.

    Each LVM is normalized by its own size (max root-to-tip vein length)
    before the top-down traversal. The result is therefore a *shape*
    distance in the geometric-morphometrics sense: invariant to a
    uniform rescaling of either input.

    Algorithm (size-normalized Selkow top-down):

    1. Synchronized traversal of both trees from their roots.
    2. At every paired non-root node, compute the absolute difference
       of normalized parent-edge lengths.
    3. If the two paired nodes have matching out-degree, their
       subtrees are isomorphic: the length difference accumulates
       into the *metric* component. Pair daughters by ``child_index``
       (1 = apical, ..., n = basal), the rooted-ordered-tree contract
       in ``docs-ferns/data_format.md``, and recurse.
    4. If out-degrees differ, positional homology cannot be extended
       beyond this node: the length difference of the parent edge
       plus the sum of normalized descendant edges on both sides
       accumulate into the *collapse* component.
    5. Return the raw costs: ``total = metric + collapse`` (the cost
       ``C``), with no per-pair divisor. Bounds are [0, Λ(T1) + Λ(T2)].

    The two components answer biologically distinct questions: the
    metric component measures proportional reallocation of edge length
    within the Maximum Common Subtree (topologically identical
    backbone); the collapse component measures the vein-length weight
    of structurally divergent subtrees plus the length difference at
    the rupture boundary.

    No joint divisor is applied. An earlier version divided both
    components by ``Λ(T1) + Λ(T2)`` to bound the result in [0, 1]; that
    per-pair divisor is what broke the triangle inequality (HCTD is a
    genuine metric without it). See DECISIONS.md and
    ``scripts/analysis/check_metric_space.py``.

    Parameters
    ----------
    G1, G2 : networkx.DiGraph
        Directed venation graphs. Each must have ``length_mm`` on every
        edge, ``child_index`` on branching edges, and a unique root.
    trace : list or None, optional
        If a list is supplied, one ``HCTDTraceEvent`` per cost step is
        appended to it, in traversal order. Amounts are raw (undivided)
        costs, so ``sum(amount)`` per component reproduces ``metric``
        and ``collapse`` exactly. ``None`` (default) leaves the
        computation and the return value unchanged.

    Returns
    -------
    HCTDPairResult
        Dataclass with fields ``metric``, ``collapse``, and ``total``
        (all in [0, Λ1 + Λ2]). ``HCTD(G, G).total == 0``;
        ``HCTD(G, k*G).total == 0`` for any positive ``k``.

    Raises
    ------
    ValueError
        If either graph has no unique root or non-positive size.
    """
    root1 = find_root(G1)
    root2 = find_root(G2)
    if root1 is None or root2 is None:
        raise ValueError(
            "hctd_pair: both graphs must have a unique root "
            "(in_degree == 0)."
        )

    size1 = lvm_size(G1)
    size2 = lvm_size(G2)
    if size1 <= 0 or size2 <= 0:
        raise ValueError(
            f"hctd_pair: non-positive LVM size (size1={size1}, "
            f"size2={size2})."
        )

    # Accumulate raw costs separately on size-normalized edge lengths.
    metric_cost = 0.0
    collapse_cost = 0.0
    stack = [(root1, root2, None, None)]

    while stack:
        n1, n2, p1, p2 = stack.pop()

        out1 = G1.out_degree(n1)
        out2 = G2.out_degree(n2)

        # Deformation cost (normalized parent-edge lengths).
        # If the paired nodes have the same out-degree (subtrees are
        # isomorphic), the length difference is metric deformation
        # within the homologous backbone. If out-degrees differ, the
        # edge leads into a rupture and its length difference is part
        # of the structural divergence (collapse).
        if p1 is not None and p2 is not None:
            w1 = G1[p1][n1]["length_mm"] / size1
            w2 = G2[p2][n2]["length_mm"] / size2
            diff = abs(w1 - w2)
            if out1 == out2:
                metric_cost += diff
                if trace is not None:
                    trace.append(HCTDTraceEvent(
                        kind="metric", node1=n1, node2=n2,
                        parent1=p1, parent2=p2,
                        out_degree1=out1, out_degree2=out2,
                        w1=w1, w2=w2, amount=diff,
                        component="metric", side="both",
                    ))
            else:
                collapse_cost += diff
                if trace is not None:
                    trace.append(HCTDTraceEvent(
                        kind="rupture_parent", node1=n1, node2=n2,
                        parent1=p1, parent2=p2,
                        out_degree1=out1, out_degree2=out2,
                        w1=w1, w2=w2, amount=diff,
                        component="collapse", side="both",
                    ))

        if out1 == out2:
            # Homology continues. Pair daughters apical -> basal by
            # their child_index (1 = apical, ..., n = basal), which the
            # extraction pipeline assigns from the midvein axis.
            children1 = ordered_children(G1, n1)
            children2 = ordered_children(G2, n2)
            for c1, c2 in zip(children1, children2):
                stack.append((c1, c2, n1, n2))
        else:
            # Homology rupture. Sum normalized descendant subtrees.
            # In strict binary trees one of the two sums is 0 (the
            # tip-side has no descendants), but the formula is general.
            sub1 = _descendant_mass(G1, n1, size1)
            collapse_cost += sub1
            if trace is not None:
                trace.append(HCTDTraceEvent(
                    kind="subtree", node1=n1, node2=n2,
                    parent1=p1, parent2=p2,
                    out_degree1=out1, out_degree2=out2,
                    w1=0.0, w2=0.0, amount=sub1,
                    component="collapse", side="1",
                ))
            sub2 = _descendant_mass(G2, n2, size2)
            collapse_cost += sub2
            if trace is not None:
                trace.append(HCTDTraceEvent(
                    kind="subtree", node1=n1, node2=n2,
                    parent1=p1, parent2=p2,
                    out_degree1=out1, out_degree2=out2,
                    w1=0.0, w2=0.0, amount=sub2,
                    component="collapse", side="2",
                ))

    # Raw costs: no joint divisor. total = metric + collapse exactly.
    metric = metric_cost
    collapse = collapse_cost
    return HCTDPairResult(
        metric=metric,
        collapse=collapse,
        total=metric + collapse,
    )


def hctd(graphs):
    """Batch HCTD computation over a collection of LVMs.

    Parameters
    ----------
    graphs : dict[str, networkx.DiGraph]
        Mapping of LVM identifier to directed graph. Every graph must
        have ``length_mm`` on edges, ``child_index`` on branching
        edges, and a unique root.

    Returns
    -------
    HCTDResult
        ``result.matrix`` is the n x n symmetric pairwise HCTD matrix;
        ``result.matrix_metric`` and ``result.matrix_collapse`` are the
        additive components (sum equals ``matrix`` element-wise);
        ``result.sizes`` is the per-LVM size_mm Series. Index and
        columns are the sorted LVM identifiers.

    Notes
    -----
    The traversal is sequential (no parallelization). For the current
    corpus scale (n <= ~100) this completes in well under a second.
    Both components are produced in the same traversal — no extra
    cost over the single-matrix variant.
    """
    names = sorted(graphs.keys())
    n = len(names)

    # Precompute sizes once per graph (used by both the matrix and
    # exposed in the result).
    sizes_mm = {name: lvm_size(graphs[name]) for name in names}

    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    matrix_metric = pd.DataFrame(
        index=names, columns=names, dtype=float
    )
    matrix_collapse = pd.DataFrame(
        index=names, columns=names, dtype=float
    )

    for i in range(n):
        matrix.at[names[i], names[i]] = 0.0
        matrix_metric.at[names[i], names[i]] = 0.0
        matrix_collapse.at[names[i], names[i]] = 0.0
        for j in range(i + 1, n):
            r = hctd_pair(graphs[names[i]], graphs[names[j]])
            matrix.at[names[i], names[j]] = r.total
            matrix.at[names[j], names[i]] = r.total
            matrix_metric.at[names[i], names[j]] = r.metric
            matrix_metric.at[names[j], names[i]] = r.metric
            matrix_collapse.at[names[i], names[j]] = r.collapse
            matrix_collapse.at[names[j], names[i]] = r.collapse

    sizes = pd.Series(sizes_mm, name="size_mm")
    sizes.index.name = "lvm_id"

    return HCTDResult(
        matrix=matrix,
        matrix_metric=matrix_metric,
        matrix_collapse=matrix_collapse,
        sizes=sizes,
    )
