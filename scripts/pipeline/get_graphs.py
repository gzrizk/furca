# =================================================
# Script:      get_graphs.py
# Description: Extracts directed venation graphs from semantic RGB masks.
#              Detects vein skeletons, snaps nodes, assigns edge lengths
#              in mm, and exports one .graphml per specimen. Supports two
#              datasets: fern trees (red/blue/magenta) and mixed graphs
#              (red/blue/magenta + yellow cross-veins).
#              Cross-veins are extracted, collapsed into sites, and added
#              as edge_type="xvein" edges; every node gets a node_role.
# INPUT:       data/01_raw/digitized/*.png            (fern)
#              data/01_raw/digitized_mixed_graph/*.png (mixed graph)
#              data/02_interim/specimen_metadata.csv
#              data/02_interim/mixed_graph_metadata.csv
# OUTPUT:      data/02_interim/graphs_graphml/*.graphml
#              data/02_interim/graphs_mixed_graph_graphml/*.graphml
#              qc_images/*.png
#              qc_images_mixed_graph/*.png
#              results/tables/graph_summary.csv
#              results/mixed_graph_graph_summary.csv
# Author:      Gonzalo Rodriguez Rizk
# Date:        2026-04-14
# =================================================

import os
import json
import yaml as _yaml
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from skimage import measure, morphology
from scipy.spatial import KDTree,  distance
from scipy.interpolate import interp1d
from PIL import Image

# --- CONFIGURATION ---
_CFG = {}
_cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
if _cfg_path.exists():
    with open(_cfg_path) as _f:
        _CFG = _yaml.safe_load(_f) or {}
_p = _CFG.get("paths", {})
_g = _CFG.get("graph_extraction", {})
_mp = _CFG.get("mixed_graph_extraction", {}).get("paths", {})

SNAP_DISTANCE = _g.get("snap_distance_px", 30)
MIN_EDGE_PX   = _g.get("min_edge_px",      3)
MIN_NODE_AREA_PX = _g.get("min_node_area_px", 4)
# Cross-vein (yellow) snapping/discard thresholds default to the backbone
# values; cross-veins can be shorter than backbone veins.
XVEIN_SNAP_DISTANCE = _g.get("xvein_snap_distance_px", SNAP_DISTANCE)
MIN_XVEIN_PX = _g.get("min_xvein_px", MIN_EDGE_PX)
# QC overlay: draw the full skeleton (faint gray) behind the graph. Off by
# default — the sampled polyline is the primary geometry; the skeleton is
# only useful for debugging snapping/spurs.
QC_SHOW_SKELETON = _g.get("qc_show_skeleton", False)

# Edge length computation method:
#   "polyline"    sample points along the skeleton curve and use the
#                 polyline length through them as the edge weight (px).
#   "pixel_count" legacy behaviour: count of skeleton pixels in the
#                 connected component (px). Underestimates diagonals by
#                 factor 1/sqrt(2). Kept for diagnostic comparison.
EDGE_LENGTH_METHOD = _g.get("edge_length_method", "polyline")

# Polyline sampling — uniform physical-space density.
# Number of curve samples per edge:
#   N = max(MIN_CURVE_POINTS,
#           int(arc_length_px * mm_per_pixel / CURVE_SAMPLING_STEP_MM))
# where mm_per_pixel is the specimen's physical scale.  Sampling density
# is therefore identical in mm across all specimens regardless of image
# resolution, so curved veins are approximated with the same fidelity
# everywhere.  If mm_per_pixel is unavailable, FALLBACK_STEP_PX is used
# (legacy pixel-space behaviour) and a WARN is emitted, since physical-space
# sampling invariance is then not preserved.
MIN_CURVE_POINTS = _g.get("min_curve_points", 2)
CURVE_SAMPLING_STEP_MM = _g.get("curve_sampling_step_mm", 0.1)
FALLBACK_STEP_PX = 15  # used when mm_per_pixel is missing or invalid


def _load_datasets():
    """
    Build the list of graph-extraction datasets from config.yaml.

    The fern dataset (``paths``) always runs. The mixed-graph dataset
    (``mixed_graph_extraction.paths``) runs only when its ``mask_dir`` is
    configured, so a fern-only checkout behaves exactly as before. Each
    entry carries its own mask/metadata/graph/QC/summary paths.

    Returns
    -------
    list of dict
        One dict per dataset with keys ``name``, ``graph_type``,
        ``mask_dir``, ``meta_file``, ``graphs_dir``, ``qc_dir``,
        ``summary_csv``, ``log_file``.
    """
    datasets = [
        {
            "name": "fern",
            "graph_type": "lvm_tree",
            "mask_dir": Path(_p.get("output_png_dir", "data/01_raw/digitized")),
            "meta_file": Path(_p.get(
                "metadata_csv", "data/02_interim/specimen_metadata.csv",
            )),
            "graphs_dir": Path(_p.get(
                "graphs_dir", "data/02_interim/graphs_graphml",
            )),
            "qc_dir": Path(_p.get("qc_dir", "qc_images")),
            "summary_csv": Path(_p.get(
                "graph_summary_csv", "results/tables/graph_summary.csv",
            )),
            "log_file": None,
        }
    ]
    if _mp.get("mask_dir"):
        datasets.append({
            "name": "mixed_graph",
            "graph_type": "mixed_graph",
            "mask_dir": Path(_mp["mask_dir"]),
            "meta_file": Path(_mp.get(
                "metadata_csv", "data/02_interim/mixed_graph_metadata.csv",
            )),
            "graphs_dir": Path(_mp.get(
                "graphs_dir", "data/02_interim/graphs_mixed_graph_graphml",
            )),
            "qc_dir": Path(_mp.get("qc_dir", "qc_images_mixed_graph")),
            "summary_csv": Path(_mp.get(
                "graph_summary_csv", "results/mixed_graph_graph_summary.csv",
            )),
            "log_file": Path(_mp.get(
                "log_file", "logs/mixed_graph_extraction_errors.log",
            )),
        })
    return datasets


# --- CORE METHODOLOGY ---

def semantic_channel_split(img_rgb):
    """
    Extract the four semantic masks from an RGB image using strict color thresholds.

    Parameters
    ----------
    img_rgb : numpy.ndarray
        RGB image array of shape (height, width, 3) with values in [0, 255].

    Returns
    -------
    mask_nodes : numpy.ndarray (bool)
        True where red channel > 250 and green, blue < 5 (nodes).
    mask_edges : numpy.ndarray (bool)
        True where blue channel > 250 and red, green < 5 (vein segments).
    mask_axis : numpy.ndarray (bool)
        True where red and blue > 250 and green < 5 (midvein axis).
    mask_xvein : numpy.ndarray (bool)
        True where red and green > 250 and blue < 5 (cross-veins).
    """
    mask_nodes = (img_rgb[:,:,0] > 250) & (img_rgb[:,:,1] < 5) & (img_rgb[:,:,2] < 5)
    mask_edges = (img_rgb[:,:,2] > 250) & (img_rgb[:,:,0] < 5) & (img_rgb[:,:,1] < 5)
    mask_axis  = (img_rgb[:,:,0] > 250) & (img_rgb[:,:,2] > 250) & (img_rgb[:,:,1] < 5)
    mask_xvein = (img_rgb[:,:,0] > 250) & (img_rgb[:,:,1] > 250) & (img_rgb[:,:,2] < 5)
    return mask_nodes, mask_edges, mask_axis, mask_xvein


def extract_node_centroids(mask_nodes, filename=""):
    """
    Identify connected components in the node mask and compute their
    centroids, discarding components smaller than ``MIN_NODE_AREA_PX``
    (stray single-pixel dots from sensor/camera noise).

    Parameters
    ----------
    mask_nodes : numpy.ndarray (bool)
        Binary mask where True indicates a node pixel.
    filename : str, optional
        Image filename, used in the warning message when components are
        discarded.

    Returns
    -------
    node_map : dict
        Mapping from node ID (int) to centroid coordinates (y, x).
    node_coords : list of tuples
        List of (y, x) coordinates for each node, in the same order as node IDs.
    tree : scipy.spatial.KDTree
        KDTree built from node_coords for fast nearest‑neighbor queries.

    Raises
    ------
    ValueError
        If fewer than 2 nodes are detected (insufficient for graph construction).
    """
    labeled_nodes, n_nodes = measure.label(mask_nodes, return_num=True)
    if n_nodes < 2:
        raise ValueError(f"Insufficient node count: detected {n_nodes} red nodes. Minimum requirement is 2.")

    regions = measure.regionprops(labeled_nodes)
    kept = [p for p in regions if p.area >= MIN_NODE_AREA_PX]
    n_discarded = len(regions) - len(kept)
    if n_discarded > 0:
        print(f"Warning [{filename}]: discarded {n_discarded} red node "
              f"component(s) smaller than {MIN_NODE_AREA_PX} px "
              f"(stray dots / sensor noise).")
    if len(kept) < 2:
        raise ValueError(f"Insufficient node count: {len(kept)} red nodes "
                         f"after filtering (< {MIN_NODE_AREA_PX} px). "
                         f"Minimum requirement is 2.")

    node_coords = []
    node_map = {}  # ID -> (y, x)

    for i, props in enumerate(kept):
        y, x = props.centroid
        node_coords.append((y, x))
        node_map[i] = (y, x)

    tree = KDTree(node_coords)
    return node_map, node_coords, tree


def skeletonize_and_label_edges(mask_edges):
    """
    Skeletonize the edge mask and label each connected skeleton branch.
    
    Parameters
    ----------
    mask_edges : numpy.ndarray (bool)
        Binary mask where True indicates a vein‑segment pixel.
        
    Returns
    -------
    labeled_edges : numpy.ndarray (int)
        Array with same shape as mask_edges, where each connected skeleton component
        has a unique integer label (background = 0).
    n_edges : int
        Number of connected skeleton components (excluding background).
    """
    skel = morphology.skeletonize(mask_edges)
    labeled_edges, n_edges = measure.label(skel, return_num=True)
    return labeled_edges, n_edges


# --- CURVE TRACING ---

def _order_skeleton_chain(component_pixels, filename=""):
    """
    Order the pixels of a single skeleton component into a sequence that
    traces the curve from one endpoint to the other.

    The skeleton is single-pixel wide (output of `morphology.skeletonize`),
    so 8-connectivity defines the chain adjacency. Two-endpoint components
    are linear chains; components with >2 endpoints carry skeletonization
    spurs and we trace the longest endpoint-to-endpoint path through the
    component (the spurs are dropped from the ordering).

    Parameters
    ----------
    component_pixels : numpy.ndarray of shape (M, 2)
        Pixel coordinates (y, x) of one connected skeleton component.
    filename : str, optional
        Image filename, used in warning messages.

    Returns
    -------
    ordered : numpy.ndarray of shape (K, 2) or None
        Ordered (y, x) coordinates from one endpoint to the other.
        K <= M because spur pixels are excluded from the path.
        Returns None if M < 2.
    """
    n = len(component_pixels)
    if n < 2:
        return None

    pixel_tree = KDTree(component_pixels)
    pairs = pixel_tree.query_pairs(r=np.sqrt(2) + 1e-6)

    chain = nx.Graph()
    chain.add_nodes_from(range(n))
    for i, j in pairs:
        weight = float(np.linalg.norm(component_pixels[i] - component_pixels[j]))
        chain.add_edge(i, j, weight=weight)

    endpoints = [i for i, deg in chain.degree() if deg == 1]

    if len(endpoints) >= 2:
        if len(endpoints) > 2:
            print(f"Warning [{filename}]: Skeleton component has "
                  f"{len(endpoints)} endpoints (spurs detected). Tracing "
                  f"the longest endpoint-to-endpoint path.")
        best_pair = None
        best_length = -1.0
        for i, ep_a in enumerate(endpoints):
            lengths = nx.single_source_dijkstra_path_length(chain, ep_a)
            for ep_b in endpoints[i + 1:]:
                if ep_b in lengths and lengths[ep_b] > best_length:
                    best_length = lengths[ep_b]
                    best_pair = (ep_a, ep_b)
        if best_pair is None:
            return None
        path_idx = nx.shortest_path(
            chain, source=best_pair[0], target=best_pair[1], weight="weight"
        )
    else:
        # No endpoints: closed loop. Trace from an arbitrary pixel to its
        # graph-farthest neighbour (covers ~half the loop).
        print(f"Warning [{filename}]: Skeleton component is a closed loop "
              f"(no degree-1 endpoints). Tracing arbitrary half-loop.")
        start = 0
        lengths = nx.single_source_dijkstra_path_length(chain, start)
        farthest = max(lengths, key=lengths.get)
        path_idx = nx.shortest_path(
            chain, source=start, target=farthest, weight="weight"
        )

    return component_pixels[path_idx]


def _polyline_length_px(ordered_coords):
    """
    Sum of Euclidean distances between consecutive (y, x) pixel
    coordinates in an ordered chain. Used both for raw arc length of an
    ordered skeleton and for the length of the polyline through curve
    samples.

    Parameters
    ----------
    ordered_coords : numpy.ndarray of shape (K, 2)
        Ordered (y, x) coordinates.

    Returns
    -------
    length_px : float
        Polyline length in pixels.
    """
    if len(ordered_coords) < 2:
        return 0.0
    diffs = np.diff(ordered_coords, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _sample_curve_points(ordered_coords, n_points):
    """
    Resample an ordered pixel chain at `n_points` equally-spaced positions
    along its cumulative arc length. The first and last samples coincide
    with the chain endpoints.

    Parameters
    ----------
    ordered_coords : numpy.ndarray of shape (K, 2)
        Ordered (y, x) coordinates of the skeleton chain.
    n_points : int
        Number of samples to take. Must be >= 2.

    Returns
    -------
    sampled : numpy.ndarray of shape (n_points, 2)
        Sampled (y, x) coordinates along the curve. dtype is float
        because samples lie at sub-pixel arc-length positions.
    """
    if n_points < 2:
        n_points = 2
    if len(ordered_coords) < 2:
        return np.asarray(ordered_coords, dtype=float)

    diffs = np.diff(ordered_coords, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_t = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_t[-1]
    if total == 0:
        return np.tile(ordered_coords[0].astype(float), (n_points, 1))

    t_query = np.linspace(0.0, total, n_points)
    fy = interp1d(cum_t, ordered_coords[:, 0], kind="linear")
    fx = interp1d(cum_t, ordered_coords[:, 1], kind="linear")
    return np.column_stack([fy(t_query), fx(t_query)])


def _chain_xy_list(sampled):
    """
    Convert resampled (y, x) skeleton points to a JSON-serializable list
    of ``[x, y]`` pairs (matching the node x/y attribute convention).

    Parameters
    ----------
    sampled : numpy.ndarray of shape (K, 2) or None
        Resampled (y, x) coordinates along a skeleton chain.

    Returns
    -------
    list of list of float or None
        ``[[x, y], ...]``; None when sampled is None.
    """
    if sampled is None:
        return None
    return [[float(x), float(y)] for y, x in sampled]


def build_undirected_graph(node_map, labeled_edges, n_edges, tree,
                           filename="", mm_per_pixel=None):
    """
    Construct an undirected graph by tracing each skeleton component as an
    ordered chain, snapping the chain endpoints to the nearest red nodes,
    and computing each edge's weight according to ``EDGE_LENGTH_METHOD``.

    Two methods are supported (see config block):
        - ``"polyline"``   : sample N points along the ordered skeleton
                             chain and use the polyline length through
                             them as the edge weight in pixels.
                             When ``mm_per_pixel`` is available, N is
                             derived from the physical arc length and
                             ``CURVE_SAMPLING_STEP_MM``, ensuring uniform
                             sampling density across all specimens.
                             Falls back to ``FALLBACK_STEP_PX`` when
                             the scale is unknown.
        - ``"pixel_count"``: legacy. Edge weight is the count of skeleton
                             pixels in the component.

    Parameters
    ----------
    node_map : dict
        Mapping from node ID to centroid (y, x).
    labeled_edges : numpy.ndarray of int
        Labeled skeleton components (background = 0).
    n_edges : int
        Number of skeleton components.
    tree : scipy.spatial.KDTree
        KDTree of node centroids for nearest-neighbour queries.
    filename : str, optional
        Image filename, used in warning messages.
    mm_per_pixel : float or None, optional
        Physical scale of the image. Used to convert pixel arc length
        to physical arc length for determining sampling density. When
        None or <= 0, falls back to FALLBACK_STEP_PX.

    Returns
    -------
    G : networkx.Graph
        Undirected graph with node attributes (x, y) and edge attributes
        ``weight`` (pixels) and ``curve_points`` (JSON string of resampled
        ``[x, y]`` points; present only in polyline mode).
    qc_overlays : dict
        Maps the unordered edge key ``(min(idx_s, idx_e), max(idx_s, idx_e))``
        to ``{"skeleton": ordered_yx, "sampled": sampled_yx_or_None}``.
        Used by ``save_qc_image``; not persisted to graphml.
    diagnostics : dict
        Aggregate stats for the INFO summary: ``{"n_edges": int,
        "sum_n_samples": int, "polyline_total_px": float,
        "pixel_count_total_px": float}``. Empty pixel_count totals when
        ``EDGE_LENGTH_METHOD == "pixel_count"``.
    """
    if EDGE_LENGTH_METHOD not in ("polyline", "pixel_count"):
        raise ValueError(
            f"Unknown EDGE_LENGTH_METHOD: {EDGE_LENGTH_METHOD!r}. "
            f"Expected 'polyline' or 'pixel_count'."
        )

    G = nx.Graph()
    for nid, pos in node_map.items():
        G.add_node(nid, x=pos[1], y=pos[0])  # store as (x, y) for plotting convenience

    qc_overlays = {}
    diagnostics = {
        "n_edges": 0,
        "sum_n_samples": 0,
        "polyline_total_px": 0.0,
        "pixel_count_total_px": 0.0,
    }

    for i in range(1, n_edges + 1):
        component_pixels = np.argwhere(labeled_edges == i)
        if len(component_pixels) < MIN_EDGE_PX:
            continue

        ordered = _order_skeleton_chain(component_pixels, filename=filename)
        if ordered is None or len(ordered) < 2:
            continue

        # Snap the two ordered-chain endpoints to red nodes
        p_start, p_end = ordered[0], ordered[-1]
        dist_s, idx_s = tree.query(p_start)
        dist_e, idx_e = tree.query(p_end)
        if dist_s > SNAP_DISTANCE or dist_e > SNAP_DISTANCE or idx_s == idx_e:
            continue

        arc_length_px = _polyline_length_px(ordered)
        pixel_count_px = float(len(component_pixels))

        if EDGE_LENGTH_METHOD == "polyline":
            if mm_per_pixel is not None and mm_per_pixel > 0:
                step_px = CURVE_SAMPLING_STEP_MM / mm_per_pixel
                if step_px < 1.0:
                    # Clamp to 1 px to prevent oversampling beyond the
                    # pixel grid when scale is anomalously large.
                    step_px = 1.0
            else:
                print(f"WARN [{filename}]: mm_per_pixel missing or invalid; "
                      f"falling back to pixel-space sampling step "
                      f"({FALLBACK_STEP_PX} px). Physical-space sampling "
                      f"invariance is not preserved.")
                step_px = FALLBACK_STEP_PX
            n_points = max(
                MIN_CURVE_POINTS,
                int(arc_length_px / step_px),
            )
            sampled = _sample_curve_points(ordered, n_points)
            weight_px = _polyline_length_px(sampled)
            diagnostics["sum_n_samples"] += n_points
        else:  # "pixel_count"
            sampled = None
            weight_px = pixel_count_px

        if weight_px <= 0.0:
            print(f"Warning [{filename}]: Edge between nodes {idx_s} and "
                  f"{idx_e} has non-positive weight ({weight_px}); skipping.")
            continue

        G.add_edge(idx_s, idx_e, weight=weight_px)
        if sampled is not None:
            G[idx_s][idx_e]["curve_points"] = json.dumps(
                _chain_xy_list(sampled),
            )
        qc_overlays[(min(idx_s, idx_e), max(idx_s, idx_e))] = {
            "skeleton": ordered,
            "sampled": sampled,
        }
        diagnostics["n_edges"] += 1
        diagnostics["polyline_total_px"] += arc_length_px
        diagnostics["pixel_count_total_px"] += pixel_count_px

    return G, qc_overlays, diagnostics


def build_xvein_records(node_map, labeled_xvein, n_xvein, tree,
                        filename="", mm_per_pixel=None):
    """
    Trace cross-vein skeleton components and snap their endpoints to red
    nodes, mirroring ``build_undirected_graph`` but returning records
    instead of inserting into a graph. Cross-veins are added to the final
    DiGraph later, after multiplicity grouping and node-role classification.

    Both endpoints of each yellow chain snap to the *same* KDTree of red
    centroids used for the blue backbone (cross-veins anchor on red nodes).

    Parameters
    ----------
    node_map : dict
        Mapping from node ID to centroid (y, x).
    labeled_xvein : numpy.ndarray of int
        Labeled skeleton components of the yellow mask (background = 0).
    n_xvein : int
        Number of yellow skeleton components.
    tree : scipy.spatial.KDTree
        KDTree of node centroids for nearest-neighbour queries.
    filename : str, optional
        Image filename, used in warning messages.
    mm_per_pixel : float or None, optional
        Physical scale; forwarded to the polyline sampling density.

    Returns
    -------
    records : list of dict
        One dict per surviving cross-vein arc: ``{"node_u", "node_v",
        "weight", "geometry"}`` where geometry is the ordered (y, x) chain.
    diagnostics : dict
        Counters ``{"n_xvein_raw", "n_xvein_discarded_short",
        "n_xvein_discarded_snap", "sum_n_samples", "polyline_total_px"}``.
    """
    diagnostics = {
        "n_xvein_raw": 0,
        "n_xvein_discarded_short": 0,
        "n_xvein_discarded_snap": 0,
        "sum_n_samples": 0,
        "polyline_total_px": 0.0,
    }
    records = []

    for i in range(1, n_xvein + 1):
        component_pixels = np.argwhere(labeled_xvein == i)
        if len(component_pixels) < MIN_XVEIN_PX:
            diagnostics["n_xvein_discarded_short"] += 1
            continue

        ordered = _order_skeleton_chain(component_pixels, filename=filename)
        if ordered is None or len(ordered) < 2:
            continue

        p_start, p_end = ordered[0], ordered[-1]
        dist_s, idx_s = tree.query(p_start)
        dist_e, idx_e = tree.query(p_end)
        if dist_s > XVEIN_SNAP_DISTANCE or dist_e > XVEIN_SNAP_DISTANCE:
            print(f"Warning [{filename}]: Cross-vein endpoint failed to snap "
                  f"(max dist {max(dist_s, dist_e):.1f} px > "
                  f"{XVEIN_SNAP_DISTANCE}); discarding component.")
            diagnostics["n_xvein_discarded_snap"] += 1
            continue
        if idx_s == idx_e:
            print(f"Warning [{filename}]: Cross-vein with both endpoints on "
                  f"node {idx_s} (self-loop); discarding.")
            diagnostics["n_xvein_discarded_snap"] += 1
            continue

        arc_length_px = _polyline_length_px(ordered)

        if EDGE_LENGTH_METHOD == "polyline":
            if mm_per_pixel is not None and mm_per_pixel > 0:
                step_px = CURVE_SAMPLING_STEP_MM / mm_per_pixel
                if step_px < 1.0:
                    step_px = 1.0
            else:
                print(f"WARN [{filename}]: mm_per_pixel missing or invalid; "
                      f"falling back to pixel-space sampling step "
                      f"({FALLBACK_STEP_PX} px). Physical-space sampling "
                      f"invariance is not preserved.")
                step_px = FALLBACK_STEP_PX
            n_points = max(MIN_CURVE_POINTS, int(arc_length_px / step_px))
            sampled = _sample_curve_points(ordered, n_points)
            weight_px = _polyline_length_px(sampled)
            diagnostics["sum_n_samples"] += n_points
        else:
            sampled = None
            weight_px = arc_length_px

        if weight_px <= 0.0:
            continue

        records.append({
            "node_u": idx_s,
            "node_v": idx_e,
            "weight": weight_px,
            "geometry": ordered,
            "sampled": sampled,
        })
        diagnostics["n_xvein_raw"] += 1
        diagnostics["polyline_total_px"] += arc_length_px

    return records, diagnostics


def group_xvein_sites(records, filename=""):
    """
    Collapse cross-vein arcs sharing the same unordered anchor pair into a
    single site with a multiplicity. This avoids parallel edges in the
    DiGraph: the only known doubled case ("lens") is two arcs diverging and
    converging between the same two red nodes.

    Group of 1 -> ``mult=1``; group of 2 -> ``mult=2`` (both geometries kept
    for QC only); group >2 -> hard error (unexpected in the dataset).

    Parameters
    ----------
    records : list of dict
        Cross-vein arc records from ``build_xvein_records``.
    filename : str, optional
        Image filename, used in error messages.

    Returns
    -------
    sites : list of dict
        One dict per site: ``{"node_u", "node_v", "mult", "weight",
        "weight_arcs", "geometries", "sampled"}``. ``weight`` is the mean
        arc weight; ``weight_arcs`` is the list of per-arc weights, one
        per contributing arc (length == mult).
    """
    groups = {}
    for rec in records:
        key = frozenset((rec["node_u"], rec["node_v"]))
        groups.setdefault(key, []).append(rec)

    sites = []
    for key, recs in groups.items():
        u, v = sorted(key)
        if len(recs) == 1:
            mult = 1
        elif len(recs) == 2:
            mult = 2
        else:
            raise ValueError(
                f"[{filename}] {len(recs)} cross-veins share the anchor "
                f"pair ({u}, {v}); >2 is unexpected. Check the annotation."
            )
        sites.append({
            "node_u": u,
            "node_v": v,
            "mult": mult,
            "weight": float(np.mean([r["weight"] for r in recs])),
            "weight_arcs": [r["weight"] for r in recs],
            "geometries": [r["geometry"] for r in recs],
            "sampled": [r.get("sampled") for r in recs],
        })
    return sites


def classify_node_roles(G, xvein_records, root_node, filename=""):
    """
    Assign each red node a role from its undirected blue degree, after
    identifying the root (which is exempt from the degree table).

    Non-root classification (derived from the "outdegree in {2,3}, never 1"
    tree-construction rule):
      - blue degree 1          -> ``tip``
      - blue degree 3 or 4     -> ``branch`` (binary / raw ternary)
      - blue degree 2          -> ``xvein_anchor`` if its yellow degree is
                                  >= 1; otherwise ``collar_anomaly`` (split
                                  blue stroke with no cross-vein).
      - blue degree 0 or >=5   -> hard error (invalid).

    A cross-vein touching a real node (tip/branch) violates the annotation
    protocol (cross-veins anchor mid-edge) and is flagged.

    Parameters
    ----------
    G : networkx.Graph
        Undirected blue backbone graph.
    xvein_records : list of dict
        Raw cross-vein arc records (for yellow degree).
    root_node : int or None
        Node ID of the root (exempt from the table).
    filename : str, optional
        Image filename, used in warning/error messages.

    Returns
    -------
    roles : dict
        ``{node_id: "root" | "tip" | "branch" | "xvein_anchor" |
        "collar_anomaly"}``.
    anomalies : list of tuple
        ``(node_id, kind)`` where kind in ``{"collar",
        "xvein_on_real_node"}``.
    """
    yellow_degree = {}
    for rec in xvein_records:
        yellow_degree[rec["node_u"]] = yellow_degree.get(rec["node_u"], 0) + 1
        yellow_degree[rec["node_v"]] = yellow_degree.get(rec["node_v"], 0) + 1

    roles = {}
    anomalies = []

    for n in G.nodes():
        blue_deg = G.degree(n)
        y_deg = yellow_degree.get(n, 0)

        if n == root_node:
            roles[n] = "root"
            continue

        if blue_deg == 1:
            roles[n] = "tip"
        elif blue_deg in (3, 4):
            roles[n] = "branch"
        elif blue_deg == 2:
            if y_deg >= 1:
                roles[n] = "xvein_anchor"
            else:
                roles[n] = "collar_anomaly"
                anomalies.append((n, "collar"))
                print(f"Warning [{filename}]: node {n} has blue degree 2 "
                      f"but no cross-vein (collar); likely a split blue "
                      f"stroke. Flagging for manual review.")
        else:
            raise ValueError(
                f"[{filename}] node {n} has blue degree {blue_deg}; expected "
                f"1, 2, 3 or 4 for a non-root node."
            )

        if roles[n] in ("tip", "branch") and y_deg > 0:
            anomalies.append((n, "xvein_on_real_node"))
            print(f"Warning [{filename}]: node {n} (role {roles[n]}) has "
                  f"yellow degree {y_deg}; a cross-vein anchors on a real "
                  f"node instead of mid-edge. Flagging annotation error.")

    return roles, anomalies


def find_root_node(mask_axis, node_map, filename):
    """
    Identify the root node as the node closest to the magenta midvein axis.
    
    Parameters
    ----------
    mask_axis : numpy.ndarray (bool)
        Binary mask where True indicates a midvein‑axis pixel.
    node_map : dict
        Mapping from node ID to centroid (y, x).
    filename : str or pathlib.Path
        Name of the image file (used for warning messages).
        
    Returns
    -------
    root_node : int or None
        Node ID of the root, or None if no axis is present and no fallback is possible.
    """
    y_ax, x_ax = np.where(mask_axis)
    axis_coords = np.column_stack((y_ax, x_ax))
    
    if len(axis_coords) == 0:
        # Fallback heuristic: If the magenta axis is absent, default to node 0.
        print(f"Warning [{filename}]: Magenta axis undetected. Defaulting to arbitrary node as root.")
        return 0
    else:
        axis_tree = KDTree(axis_coords)
        min_dist = np.inf
        root_node = None
        for nid, (y, x) in node_map.items():
            d, _ = axis_tree.query([y, x])
            if d < min_dist:
                min_dist = d
                root_node = nid
        return root_node


def convert_to_dag(G, root_node, filename=""):
    """
    Convert an undirected graph into a Directed Acyclic Graph (DAG) by imposing
    a BFS tree rooted at `root_node`.
    
    Parameters
    ----------
    G : networkx.Graph
        Undirected graph with node attributes (x, y) and edge attributes
        (weight, curve_points).
    root_node : int or None
        Node ID to use as the root. If None or not in G, a fallback directed graph
        is returned.
    filename : str, optional
        Name of the image file (used for warning messages).
        
    Returns
    -------
    DiG : networkx.DiGraph
        Directed acyclic graph (BFS tree) if root_node is valid and reachable;
        otherwise a fully directed version of G.
    """
    if root_node is not None and root_node in G:
        DiG = nx.bfs_tree(G, source=root_node)
        # Propagate node attributes
        for n in DiG.nodes():
            DiG.nodes[n].update(G.nodes[n])
        # Propagate edge attributes (weight, curve_points)
        for u, v in DiG.edges():
            if G.has_edge(u, v):
                DiG[u][v].update(G[u][v])
        return DiG
    else:
        # Fallback: graph is disconnected or root is isolated.
        # Return a fully directed graph to preserve topological data.
        if filename:
            print(f"Warning [{filename}]: Root node {root_node} not in graph or graph disconnected. Using fully directed graph as fallback.")
        DiG = G.to_directed()
        return DiG


def midvein_vector(mask_axis):
    """
    Return the midvein direction vector ``u0 = topmost - bottommost`` of the
    magenta pixels (basal -> apical; image coordinates, y increasing
    downward so topmost = minimal y).

    Parameters
    ----------
    mask_axis : numpy.ndarray (bool)
        Binary mask of midvein-axis (magenta) pixels.

    Returns
    -------
    numpy.ndarray or None
        ``(x, y)`` direction vector, or None when the mask is empty or the
        axis is degenerate (zero length).
    """
    y_ax, x_ax = np.where(mask_axis)
    if len(y_ax) == 0:
        return None
    top = np.array([x_ax[int(np.argmin(y_ax))],
                    y_ax[int(np.argmin(y_ax))]], dtype=float)
    bot = np.array([x_ax[int(np.argmax(y_ax))],
                    y_ax[int(np.argmax(y_ax))]], dtype=float)
    u0 = top - bot
    if np.dot(u0, u0) == 0.0:
        return None
    return u0


def _cross2d(a, b):
    """2D cross product ``a.x * b.y - a.y * b.x`` for (x, y) vectors."""
    return a[0] * b[1] - a[1] * b[0]


def _angle_with(u0, v):
    """Angle in radians between vectors ``u0`` and ``v`` (0 to pi)."""
    denom = np.linalg.norm(u0) * np.linalg.norm(v)
    if denom == 0.0:
        return None
    cos = np.clip(np.dot(u0, v) / denom, -1.0, 1.0)
    return float(np.arccos(cos))


def assign_child_index(DiG, u0, filename=""):
    """
    Assign a ``child_index`` to every branching edge of a rooted ordered
    tree, ranking daughters by their angle with the midvein axis.

    The reference direction is the midvein ``u0`` (basal -> apical), the
    biological apical/basal axis of the lamina, applied to every node
    (there is no per-node parent arc and no special case for the root).
    For each branching node ``P`` and daughter ``C``, the angle is measured
    between ``u0`` and the vector ``v = C - P``. Daughters are ranked by
    increasing angle, so ``child_index`` is 1 for the apicalmost edge,
    2, ..., n for the increasingly basal ones. 

    Daughters are compared at a common length, set by the shortest daughter
    (straight node-to-node distance). A longer daughter is judged by its
    initial direction over that length rather than by its endpoint, so a
    basal branch that curves upwards is not mistaken for an apical one.

    Only branching nodes (two or more children) carry ``child_index``: a
    single child has no rank, so no index is assigned.

    Parameters
    ----------
    DiG : networkx.DiGraph
        Directed rooted tree (BFS tree from ``convert_to_dag``).
    u0 : numpy.ndarray
        Midvein direction (x, y), basal -> apical.
    filename : str, optional
        Image filename, used in warning messages.
    """
    u0 = np.asarray(u0, dtype=float)
    for P in DiG.nodes():
        children = list(DiG.successors(P))
        if len(children) < 2:
            continue

        angles = {}
        vectors = {}
        lengths = {}
        for c in children:
            v = np.array([
                DiG.nodes[c]["x"] - DiG.nodes[P]["x"],
                DiG.nodes[c]["y"] - DiG.nodes[P]["y"],
            ], dtype=float)
            vectors[c] = v
            lengths[c] = float(np.linalg.norm(v))

        shortest = min(lengths.values())
        for c in children:
            angles[c] = _angle_with(u0, vectors[c] * (shortest / lengths[c]))

        ranked = sorted(children, key=lambda c: angles[c])
        for rank, c in enumerate(ranked, start=1):
            DiG[P][c]["child_index"] = rank


def classify_hemilamina(DiG, u0, root_node, filename=""):
    """
    Classify the LVM as acroscopic or basiscopic (optional step).

    The hemilamina is a relational property between the graph and the
    midvein, not an intrinsic property of the graph. It is read from the
    sign of the cross product between the midvein vector ``u0`` (basal ->
    apical) and the vector from the root to its first child:

        cross > 0  -> acroscopic
        cross < 0  -> basiscopic

    This step is independent of ``assign_child_index``; callers that do not
    work with hemilaminas can simply omit the call.

    Parameters
    ----------
    DiG : networkx.DiGraph
        Directed rooted tree (BFS tree from ``convert_to_dag``).
    u0 : numpy.ndarray
        Midvein direction (x, y), basal -> apical.
    root_node : int or None
        Node ID of the root.
    filename : str, optional
        Image filename, used in warning messages.

    Returns
    -------
    str or None
        ``"acroscopic"`` or ``"basiscopic"``, or None when it cannot be
        resolved (no valid root or no first child).
    """
    if root_node is None or root_node not in DiG:
        print(f"Warning [{filename}]: no valid root; cannot classify "
              f"hemilamina.")
        return None
    children = list(DiG.successors(root_node))
    if not children:
        print(f"Warning [{filename}]: root has no child; cannot classify "
              f"hemilamina.")
        return None

    first = children[0]
    v = np.array([
        DiG.nodes[first]["x"] - DiG.nodes[root_node]["x"],
        DiG.nodes[first]["y"] - DiG.nodes[root_node]["y"],
    ], dtype=float)
    cross = _cross2d(np.asarray(u0, dtype=float), v)
    return "acroscopic" if cross > 0.0 else "basiscopic"


def graph_from_mask(image_path, mm_per_pixel=None, graph_type="lvm_tree"):
    """
    Convert a semantic PNG mask into a Directed Acyclic Graph (DAG) using
    KDTree snapping and BFS distal routing. When a yellow (cross-vein)
    channel is present, cross-veins are extracted, collapsed into sites,
    and added to the DAG as ``edge_type="xvein"`` edges carrying a
    ``protocol_violation`` flag (True when the cross-vein anchors on a real
    tip/branch node instead of mid-edge); every node gains a ``node_role``
    attribute. Backbone and cross-vein edges store their resampled geometry
    as a ``curve_points`` JSON string (polyline mode only).

    For ``graph_type="lvm_tree"`` (fern trees) the graph is exported as a
    *rooted ordered tree*: every branching edge receives a ``child_index``
    (1 = apical, ..., n = basal) ranking daughters by their angle with the
    midvein vector (see ``assign_child_index``). The hemilamina
    (acroscopic/basiscopic) is an optional, decoupled step
    (see ``classify_hemilamina``). Mixed graphs are left unordered.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the semantic mask PNG file.
    mm_per_pixel : float or None, optional
        Physical scale of the image. Forwarded to ``build_undirected_graph``
        and ``build_xvein_records`` for uniform physical-space polyline
        sampling density. When None, falls back to a fixed pixel step.
    graph_type : str, optional
        ``"lvm_tree"`` (fern; ordered child assignment enabled) or
        ``"mixed_graph"`` (cross-veins; no child ordering).

    Returns
    -------
    dict
        ``DiG`` (networkx.DiGraph, backbone tree + xvein edges),
        ``img_rgb`` (numpy.ndarray), ``root_node`` (int or None),
        ``labeled_edges`` (numpy.ndarray), ``qc_overlays`` (dict),
        ``diagnostics`` (dict, backbone counters), ``node_roles`` (dict),
        ``anomalies`` (list of (node_id, kind)), ``xvein_sites`` (list of
        dict), ``xvein_diag`` (dict, cross-vein counters),
        ``hemilamina`` (str or None).

    Raises
    ------
    ValueError
        If the mask contains fewer than 2 red nodes, or a non-root node has
        an invalid blue degree, or >2 cross-veins share an anchor pair.
    """
    filename = Path(image_path).name

    # 1. Image ingestion - using PIL to ignore EXIF orientation
    pil_img = Image.open(image_path)
    img_rgb = np.array(pil_img)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]

    # 2. Semantic channel separation (4 channels)
    mask_nodes, mask_edges, mask_axis, mask_xvein = semantic_channel_split(
        img_rgb,
    )

    # 3. Node localization
    node_map, node_coords, tree = extract_node_centroids(
        mask_nodes, filename=filename,
    )

    # 4. Backbone edge extraction via morphological skeletonization
    labeled_edges, n_edges = skeletonize_and_label_edges(mask_edges)

    # 5. Build undirected blue backbone graph (also yields curve geometry
    #    for QC and aggregate diagnostics for the end-of-pipeline summary)
    G, qc_overlays, diagnostics = build_undirected_graph(
        node_map, labeled_edges, n_edges, tree,
        filename=filename, mm_per_pixel=mm_per_pixel,
    )

    # 6. Cross-vein extraction (yellow channel). An empty yellow mask
    #    (fern images) yields zero records with no config flag needed.
    labeled_xvein, n_xvein = skeletonize_and_label_edges(mask_xvein)
    xvein_records, xvein_diag = build_xvein_records(
        node_map, labeled_xvein, n_xvein, tree,
        filename=filename, mm_per_pixel=mm_per_pixel,
    )

    # 7. Root node identification (magenta axis)
    root_node = find_root_node(mask_axis, node_map, filename)

    # 8. Node-role classification. Root is identified first and is exempt
    #    from the blue-degree table (a degree-2 root is a valid immediate
    #    bifurcation, not an xvein_anchor).
    node_roles, anomalies = classify_node_roles(
        G, xvein_records, root_node, filename=filename,
    )

    # 9. Collapse cross-vein arcs into sites (multiplicity grouping)
    xvein_sites = group_xvein_sites(xvein_records, filename=filename)

    # 10. DAG construction (BFS rooted). node_role is set on G so that
    #     convert_to_dag propagates it onto the directed tree nodes.
    for n, role in node_roles.items():
        G.nodes[n]["node_role"] = role
    DiG = convert_to_dag(G, root_node, filename=filename)

    # 10b. Order children and classify hemilamina (fern trees only).
    #      child_index ranks daughters by their angle with the midvein
    #      vector u0 (1 = apical, ..., n = basal); the hemilamina is an
    #      optional, decoupled step. Neither is stored on the graph.
    hemilamina = None
    u0 = None
    if graph_type == "lvm_tree":
        u0 = midvein_vector(mask_axis)
        if u0 is None:
            print(f"Warning [{filename}]: midvein axis unavailable; child "
                  f"ordering and hemilamina skipped.")
        else:
            assign_child_index(DiG, u0, filename=filename)
            # Optional: hemilamina classification. Comment out this call
            # for structures without mirrored hemilaminas.
            hemilamina = classify_hemilamina(
                DiG, u0, root_node, filename=filename,
            )

    # 11. Add cross-vein edges. Direction (min -> max node id) has NO
    #     topological meaning: only the presence of the connection does.
    #     A cross-vein whose endpoint falls on a real (tip/branch) node
    #     violates the annotation protocol (cross-veins anchor mid-edge).
    for site in xvein_sites:
        protocol_violation = (
            node_roles.get(site["node_u"]) in ("tip", "branch")
            or node_roles.get(site["node_v"]) in ("tip", "branch")
        )
        if protocol_violation:
            print(f"Warning [{filename}]: cross-vein between "
                  f"{site['node_u']} and {site['node_v']} anchors on a "
                  f"real node; flagging protocol violation.")
        edge_attrs = {
            "edge_type": "xvein",
            "mult": site["mult"],
            "weight": site["weight"],
            "weight_arcs": json.dumps(site["weight_arcs"]),
            "protocol_violation": protocol_violation,
        }
        chains = [_chain_xy_list(s) for s in site["sampled"] if s is not None]
        if chains:
            edge_attrs["curve_points"] = json.dumps(chains)
        DiG.add_edge(site["node_u"], site["node_v"], **edge_attrs)

    return {
        "DiG": DiG,
        "img_rgb": img_rgb,
        "root_node": root_node,
        "labeled_edges": labeled_edges,
        "qc_overlays": qc_overlays,
        "diagnostics": diagnostics,
        "node_roles": node_roles,
        "anomalies": anomalies,
        "xvein_sites": xvein_sites,
        "xvein_diag": xvein_diag,
        "hemilamina": hemilamina,
    }


def _draw_scale_bar(ax, mm_per_pixel, pos=(0.03, 0.03)):
    """
    Draw a 1 mm scale bar (white with black border) in the bottom-left
    corner.  Bar height, border width, and text offset are sized as
    fractions of the axes range so the bar looks proportionally the same
    across images of any size.

    Suppressed when *mm_per_pixel* is None, NaN, ≤ 0, or > 1.0
    (the > 1.0 threshold excludes one outlier at 25.4 mm/px).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    mm_per_pixel : float or None
        Physical scale. When None or invalid the function returns early.
    pos : tuple of float, optional
        Relative (x, y) position within the axes, in fraction of axis range.
    """
    if mm_per_pixel is None or not np.isfinite(mm_per_pixel):
        return
    if mm_per_pixel <= 0 or mm_per_pixel > 1.0:
        return

    REL_BAR_HEIGHT = 0.0125   # 1.25 % of axes height
    REL_TEXT_OFFSET = 0.005   # 0.5 % of axes height

    bar_px = 1.0 / mm_per_pixel

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    x0 = xlim[0] + pos[0] * x_range
    y0 = ylim[0] + pos[1] * y_range

    bar_height = REL_BAR_HEIGHT * y_range
    text_offset = REL_TEXT_OFFSET * y_range

    # Solid black bar
    bar = Rectangle(
        (x0, y0), bar_px, bar_height,
        facecolor='black', edgecolor='none',
    )
    ax.add_patch(bar)

    # Label centred above the bar
    label_y = y0 + bar_height + text_offset
    ax.text(
        x0 + bar_px / 2.0, label_y, "1 mm",
        color='black', fontsize=8,
        ha='center', va='bottom',
    )


def _point_at_fraction(poly, fraction):
    """Return (point, unit tangent) at `fraction` of a polyline's length."""
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total == 0.0:
        return poly[0], np.array([0.0, 0.0])
    target = fraction * total
    acc = 0.0
    for i, length in enumerate(seg_len):
        if acc + length >= target or i == len(seg_len) - 1:
            local = (target - acc) / length if length > 0 else 0.0
            point = poly[i] + local * seg[i]
            direction = seg[i] / length if length > 0 else np.array([0.0, 0.0])
            return point, direction
        acc += length
    return poly[-1], seg[-1] / seg_len[-1]


def save_qc_image(img_rgb, DiG, root_node, filename, out_qc,
                  labeled_edges=None, qc_overlays=None,
                  node_roles=None, anomalies=None, xvein_sites=None,
                  hemilamina=None):
    """
    Generate a comparative Quality Control (QC) image illustrating
    directional flow and curve geometry.

    The grayscale annotation is shown at full opacity in the right panel,
    with the extracted graph overlaid as a single thin orange polyline
    whose extremities are snapped to the node centroids (matching the
    exported graph edges). Every node is marked with a small dot coloured
    by role, so that the overlay reads as a graph (vertices plus edges):
    the annotation already marks the node positions, and the dot confirms
    that each vertex was recovered. The root is marked in white.

    Right panel layers:
        1. RGB mask, full opacity
        2. Full skeleton (faint gray) — only when ``QC_SHOW_SKELETON`` is on
        3. Polyline through the sampled backbone (orange)
        4. Cross-veins (Tol yellow polylines) — mixed graphs only
        5. Nodes (black dots; root white)
        6. Anomaly nodes (red alert ring)

    Color palette: Paul Tol's qualitative scheme (colourblind-safe).

    The backbone polyline is skipped automatically when
    ``EDGE_LENGTH_METHOD`` is ``"pixel_count"`` (no samples in that mode).

    Parameters
    ----------
    img_rgb : numpy.ndarray
        Source RGB image.
    DiG : networkx.DiGraph
        Directed graph with node attributes (x, y, node_role) and edge
        attributes (weight, and edge_type="xvein" for cross-veins).
    root_node : int or None
        Node ID of the root, or None if unidentified.
    filename : str
        Stem used for the output file: ``{filename}_QC.png``.
    out_qc : pathlib.Path
        Output directory for the QC image.
    labeled_edges : numpy.ndarray of int, optional
        Labeled skeleton components from ``skeletonize_and_label_edges``.
        When provided, the full skeleton is overlaid for visual reference.
    qc_overlays : dict, optional
        ``{(min_idx, max_idx): {"skeleton": ordered_yx, "sampled":
        sampled_yx_or_None}}`` from ``build_undirected_graph``.
    node_roles : dict, optional
        ``{node_id: role}`` from ``classify_node_roles``.
    anomalies : list of tuple, optional
        ``(node_id, kind)`` from ``classify_node_roles``.
    xvein_sites : list of dict, optional
        Collapsed cross-vein sites from ``group_xvein_sites``.
    """
    plt.figure(figsize=(12, 6))

    # Left panel: Input semantic mask
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"{filename}\nRGB semantic annotation")
    plt.axis('off')

    # Right panel: extracted graph overlaid on the annotation. The mask is
    # shown at full opacity and the graph is drawn as a single thin dark
    # polyline, so the reader can verify that the extracted geometry runs
    # along the annotation instead of the graph covering it.
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)

    legend_handles = []

    # Optional faint full skeleton (off by default)
    if QC_SHOW_SKELETON and labeled_edges is not None:
        skel_yx = np.argwhere(labeled_edges > 0)
        if len(skel_yx) > 0:
            plt.scatter(skel_yx[:, 1], skel_yx[:, 0],
                        s=1, c='#999999', alpha=0.4, marker='.',
                        linewidths=0, label='Skeleton')
            legend_handles.append(plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='#999999', markersize=6,
                label='Skeleton'
            ))

    pos = {n: (DiG.nodes[n]['x'], DiG.nodes[n]['y']) for n in DiG.nodes()}

    # Polyline through the sampled backbone, snapped to the node centroids.
    # The polyline extremities are prepended/appended with the centroid of
    # the parent/child nodes, so the drawn edge matches the exported graph
    # edge instead of stopping where the raw skeleton happened to end.
    if qc_overlays and EDGE_LENGTH_METHOD == "polyline":
        polyline_drawn = False
        for (a, b), ov in qc_overlays.items():
            sampled = ov.get("sampled")
            if sampled is None or len(sampled) < 2:
                continue
            pts_xy = sampled[:, [1, 0]]
            a, b = int(a), int(b)
            p_start = np.asarray(pos[a], dtype=float)
            p_end = np.asarray(pos[b], dtype=float)
            pts = np.vstack((p_start[None, :], pts_xy, p_end[None, :]))
            plt.plot(pts[:, 0], pts[:, 1], color='#FF7F00', lw=1.0,
                     alpha=1.0, zorder=8)
            polyline_drawn = True
        if polyline_drawn:
            legend_handles.append(plt.Line2D(
                [0], [0], color='#FF7F00', lw=1.5, label='Extracted graph'
            ))

    # Daughter order (child_index): each branching edge is annotated with
    # its rank (1 = apical, ..., n = basal) near its parent end, nudged
    # off the stroke, so the QC image shows the ordering the pipeline
    # assigns rather than only asserting it.
    if qc_overlays and EDGE_LENGTH_METHOD == "polyline":
        label_fraction = 0.3
        label_offset_pt = 3.0
        for u, v, data in DiG.edges(data=True):
            cidx = data.get("child_index")
            if cidx is None or u not in pos or v not in pos:
                continue
            ov = qc_overlays.get((min(int(u), int(v)), max(int(u), int(v))))
            if ov is None:
                continue
            sampled = ov.get("sampled")
            if sampled is None or len(sampled) < 2:
                continue
            pts_xy = sampled[:, [1, 0]]
            pts = np.vstack((
                np.asarray(pos[u], dtype=float)[None, :],
                pts_xy,
                np.asarray(pos[v], dtype=float)[None, :],
            ))
            point, direction = _point_at_fraction(pts, label_fraction)
            normal = np.array([-direction[1], direction[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            plt.annotate(
                str(int(cidx)), xy=point,
                xytext=label_offset_pt * normal,
                textcoords="offset points",
                color="#000000", fontsize=6,
                ha="center", va="center", zorder=12,
            )

    # Cross-veins (yellow). Both geometries of a mult=2 site are
    # drawn so the "lens" is auditable.
    if xvein_sites:
        xvein_drawn = False
        for site in xvein_sites:
            for geometry in site["geometries"]:
                if geometry is None or len(geometry) < 2:
                    continue
                ys, xs = geometry[:, 0], geometry[:, 1]
                plt.plot(xs, ys, color='#CCBB44', lw=1.5, alpha=0.9,
                         zorder=8)
                xvein_drawn = True
        if xvein_drawn:
            legend_handles.append(plt.Line2D(
                [0], [0], color='#CCBB44', lw=1.5, label='Cross-vein'
            ))

    # Every node as a small dot coloured by role, so the overlay reads as
    # a graph (vertices plus edges). The root is drawn white with a thin
    # black outline so it stays legible over the annotation.
    if pos:
        tips = [pos[n] for n in DiG.nodes() if n != root_node]
        if tips:
            tx = [p[0] for p in tips]
            ty = [p[1] for p in tips]
            plt.scatter(tx, ty, s=2, c='#000000', marker='o',
                        linewidths=0, zorder=9)
            legend_handles.append(plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='#000000', markersize=2,
                label='Nodes'
            ))
        if root_node is not None and root_node in pos:
            plt.scatter([pos[root_node][0]], [pos[root_node][1]],
                        s=2, facecolors='#FFFFFF', edgecolors='#000000',
                        linewidths=0.5, zorder=10)
            legend_handles.append(plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='#FFFFFF', markeredgecolor='#000000',
                markersize=2, label='Root'
            ))

    # Hemilamina side label (acroscopic / basiscopic), placed next to the
    # root node so the audit can read the classification straight off the
    # overlay.
    if hemilamina is not None and root_node is not None and root_node in pos:
        label = "acro" if hemilamina == "acroscopic" else "basi"
        plt.annotate(
            label, xy=pos[root_node],
            xytext=(6, -6), textcoords="offset points",
            color="#000000", fontsize=8,
            ha="left", va="top", zorder=12,
        )

    # Anomaly nodes (red alert ring)
    if anomalies:
        anomaly_nodes = {n for n, _ in anomalies}
        axs = [pos[n][0] for n in anomaly_nodes if n in pos]
        ays = [pos[n][1] for n in anomaly_nodes if n in pos]
        if axs:
            plt.scatter(axs, ays, s=200, facecolors='none',
                        edgecolors='#CC3311', linewidths=2, zorder=11)
            legend_handles.append(plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor='none', markeredgecolor='#CC3311',
                markersize=10, label='Anomaly (review)'
            ))

    n_info = DiG.number_of_nodes()
    e_info = DiG.number_of_edges()
    plt.title(
        f"Extracted venation tree (directed, rooted)\n"
        f"{n_info} nodes, {e_info} edges, length: {EDGE_LENGTH_METHOD}"
    )

    mm_per_pixel = DiG.graph.get("mm_per_pixel")
    _draw_scale_bar(plt.gca(), mm_per_pixel)

    plt.axis('off')
    if legend_handles:
        plt.legend(handles=legend_handles, loc='lower right',
                   fontsize=7, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(out_qc / f"{filename}_QC.png", dpi=100)
    plt.close()

def main():
    """Run graph extraction for every configured dataset."""
    for ds in _load_datasets():
        _process_dataset(ds)


def _process_dataset(ds):
    """Run the extraction pipeline for a single dataset.

    Parameters
    ----------
    ds : dict
        Dataset descriptor from ``_load_datasets``.
    """
    name = ds["name"]
    graph_type = ds["graph_type"]
    mask_dir = ds["mask_dir"]
    meta_file = ds["meta_file"]
    graphs_dir = ds["graphs_dir"]
    qc_dir = ds["qc_dir"]
    summary_csv = ds["summary_csv"]
    log_file = ds["log_file"]

    for p in [graphs_dir, qc_dir, summary_csv.parent]:
        p.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    stats_list = []
    error_log = []
    side_map = {}

    # Load metadata to inject physical scale factors
    if meta_file.exists():
        df_meta = pd.read_csv(meta_file)
        scale_map = dict(zip(df_meta["filename_png"], df_meta["mm_per_pixel"]))
    else:
        scale_map = {}

    print(f"Initiating topological extraction pipeline [{name}] on: {mask_dir}")

    processed_count = 0
    error_count = 0

    # Aggregate diagnostics across all processed masks (printed at end)
    agg_n_edges = 0
    agg_sum_n_samples = 0
    agg_polyline_total_px = 0.0
    agg_pixel_count_total_px = 0.0

    for mask_file in sorted(mask_dir.glob("*.png")):
        try:
            scale = scale_map.get(mask_file.name, -1)

            # 1. Execute mask parsing (physical scale passed for uniform
            #    physical-space polyline sampling density)
            result = graph_from_mask(
                mask_file, mm_per_pixel=scale if scale > 0 else None,
                graph_type=graph_type,
            )
            DiG = result["DiG"]
            img_rgb = result["img_rgb"]
            root_node = result["root_node"]
            labeled_edges = result["labeled_edges"]
            qc_overlays = result["qc_overlays"]
            diagnostics = result["diagnostics"]
            node_roles = result["node_roles"]
            anomalies = result["anomalies"]
            xvein_sites = result["xvein_sites"]
            xvein_diag = result["xvein_diag"]
            hemilamina = result.get("hemilamina")

            # 2. Inject physical scale metrics (if available)
            DiG.graph["mm_per_pixel"] = scale
            DiG.graph["filename"] = mask_file.name
            DiG.graph["edge_length_method"] = EDGE_LENGTH_METHOD
            DiG.graph["graph_type"] = graph_type
            DiG.graph["has_xveins"] = bool(xvein_sites)

            # Anomaly counts (gate for downstream normalization): collar
            # nodes (split blue stroke) and cross-veins touching real nodes.
            n_collar_anomalies = sum(
                1 for r in node_roles.values() if r == "collar_anomaly"
            )
            n_xvein_protocol_violations = sum(
                1 for _, _, d in DiG.edges(data=True)
                if d.get("protocol_violation")
            )
            has_anomalies = (
                n_collar_anomalies > 0 or n_xvein_protocol_violations > 0
            )
            DiG.graph["has_anomalies"] = has_anomalies

            if scale > 0:
                for u, v, d in DiG.edges(data=True):
                    d["length_mm"] = d["weight"] * scale

            # 3. Export graph topology to GraphML format
            nx.write_graphml(DiG, graphs_dir / f"{mask_file.stem}.graphml")

            # Collect the hemilamina side (axiomatic cross-product sign).
            # It is not a graph property, so it is not written to the
            # GraphML; it is persisted to the metadata CSV instead.
            if hemilamina is not None:
                side_map[mask_file.name] = hemilamina

            # 4. Generate Quality Control (QC) visualizations
            save_qc_image(
                img_rgb, DiG, root_node, mask_file.stem, qc_dir,
                labeled_edges=labeled_edges, qc_overlays=qc_overlays,
                node_roles=node_roles, anomalies=anomalies,
                xvein_sites=xvein_sites, hemilamina=hemilamina,
            )

            # 5. Compile topological statistics
            n_xvein_sites = len(xvein_sites)
            n_xvein_raw = xvein_diag["n_xvein_raw"]
            n_mult2 = sum(1 for s in xvein_sites if s["mult"] == 2)
            n_anchor = sum(
                1 for r in node_roles.values() if r == "xvein_anchor"
            )
            n_anomalies = len(anomalies)

            # Consistency check: every mult=2 site contributes one extra
            # raw arc over the number of sites.
            if n_xvein_raw != n_xvein_sites + n_mult2:
                print(
                    f"WARNING [{mask_file.name}]: n_xvein_raw ({n_xvein_raw})"
                    f" != n_xvein_sites ({n_xvein_sites}) + n_mult2 "
                    f"({n_mult2})."
                )

            stats_list.append({
                "filename": mask_file.name,
                "nodes": DiG.number_of_nodes(),
                "edges": DiG.number_of_edges(),
                "n_xvein_sites": n_xvein_sites,
                "n_xvein_edges_raw": n_xvein_raw,
                "n_xvein_anchor_nodes": n_anchor,
                "n_anomalias_flagged": n_anomalies,
                "n_collar_anomalies": n_collar_anomalies,
                "n_xvein_protocol_violations": n_xvein_protocol_violations,
                "has_anomalies": has_anomalies,
                "scale_used": scale,
                "edge_length_method": EDGE_LENGTH_METHOD,
            })

            agg_n_edges += diagnostics["n_edges"]
            agg_sum_n_samples += diagnostics["sum_n_samples"]
            agg_polyline_total_px += diagnostics["polyline_total_px"]
            agg_pixel_count_total_px += diagnostics["pixel_count_total_px"]

            processed_count += 1
            print(
                f"Processed {mask_file.name}: {DiG.number_of_nodes()} nodes, "
                f"{n_xvein_sites} xvein site(s), {n_anomalies} anomalies."
            )

        except Exception as e:
            print(f"Error processing {mask_file.name}: {e}")
            error_log.append(f"{mask_file.name}: {e}")
            error_count += 1

    if stats_list:
        pd.DataFrame(stats_list).to_csv(summary_csv, index=False)
    if error_log and log_file is not None:
        with open(log_file, "w") as f:
            f.write("\n".join(error_log) + "\n")

    # Persist the hemilamina side to the metadata CSV. The axiomatic
    # cross-product classification is the single source of truth, so
    # existing values are replaced (purging any legacy assignment);
    # masks without a side are skipped (no classification available).
    if side_map and meta_file.exists():
        df_meta = pd.read_csv(meta_file)
        if "side" not in df_meta.columns:
            df_meta["side"] = pd.NA
        df_meta["side"] = df_meta["filename_png"].map(side_map).fillna(df_meta["side"])
        df_meta.to_csv(meta_file, index=False)
        n_acro = sum(1 for s in side_map.values() if s == "acroscopic")
        n_basi = sum(1 for s in side_map.values() if s == "basiscopic")
        print(f"Hemilamina side written to: {meta_file}")
        print(f"  acroscopic: {n_acro} | basiscopic: {n_basi} "
              f"(skipped: {len(df_meta) - len(side_map)})")

    print("-" * 30)
    print(f"Pipeline execution complete [{name}].")
    print(f"Total files processed successfully: {processed_count}")
    print(f"Total processing errors: {error_count}")

    print(f"Edge length method: {EDGE_LENGTH_METHOD}")
    print(f"Total edges across all graphs: {agg_n_edges}")
    if agg_n_edges > 0:
        if EDGE_LENGTH_METHOD == "polyline" and agg_sum_n_samples > 0:
            mean_n = agg_sum_n_samples / agg_n_edges
            print(f"Mean curve samples per edge: {mean_n:.1f}")
        if agg_pixel_count_total_px > 0:
            # arc length (true Euclidean traversal) vs pixel count.
            # A ratio > 1 means pixel count was underestimating arc length
            # (diagonals counted as 1 px instead of sqrt(2) px).
            ratio = agg_polyline_total_px / agg_pixel_count_total_px
            print(f"Mean arc-length / pixel-count ratio: {ratio:.3f}")
    print(f"Output topological models saved to: {graphs_dir}")


if __name__ == "__main__":
    main()
