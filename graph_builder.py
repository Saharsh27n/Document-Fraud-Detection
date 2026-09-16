"""
Stage 2 — Spatial graph builder.

build_spatial_graph(layout_elements, image_shape) -> networkx.Graph
Visualises graph to output/graphs/.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBOURS = 3
ALIGN_TOL = 0.02  # 2% of page dimension

TYPE_COLORS = {
    "text": "#4a90d9",
    "title": "#e8552c",
    "table": "#2ca02c",
    "figure": "#9467bd",
    "list": "#8c564b",
    "unknown": "#7f7f7f",
}


def build_spatial_graph(
    layout_elements: list[dict],
    image_shape: tuple[int, int],
    image_stem: Optional[str] = None,
) -> nx.Graph:
    """
    Build a spatial graph from detected layout elements.

    Args:
        layout_elements: output of extract_layout()
        image_shape: (height, width) of the original image
        image_stem: if given, graph visualisation is saved to output/graphs/

    Returns:
        networkx.Graph with node/edge attributes
    """
    H, W = image_shape
    if not layout_elements:
        return nx.Graph()

    G = nx.Graph()

    # --- Add nodes ---
    for i, elem in enumerate(layout_elements):
        x1, y1, x2, y2 = elem["bbox"]
        # Normalise to [0, 1]
        nx1, ny1, nx2, ny2 = x1 / W, y1 / H, x2 / W, y2 / H
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        nw = nx2 - nx1
        nh = ny2 - ny1
        area_ratio = nw * nh  # already normalised

        G.add_node(i, **{
            "type": elem.get("type", "unknown"),
            "bbox": (nx1, ny1, nx2, ny2),
            "centroid": (cx, cy),
            "norm_width": nw,
            "norm_height": nh,
            "area_ratio": area_ratio,
            "confidence": elem.get("confidence", 1.0),
        })

    nodes = list(G.nodes(data=True))
    n = len(nodes)

    # --- kNN edges ---
    centroids = np.array([d["centroid"] for _, d in nodes])
    if n > 1:
        from scipy.spatial.distance import cdist
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        k = min(K_NEIGHBOURS, n - 1)
        for i in range(n):
            knn_idx = np.argsort(dists[i])[:k]
            for j in knn_idx:
                if not G.has_edge(i, j):
                    G.add_edge(i, j, dist=float(dists[i, j]), align="knn")

    # --- Alignment edges ---
    for i in range(n):
        _, di = nodes[i]
        ci = di["centroid"]
        for j in range(i + 1, n):
            _, dj = nodes[j]
            cj = dj["centroid"]
            if G.has_edge(i, j):
                continue
            horiz_aligned = abs(ci[1] - cj[1]) < ALIGN_TOL  # same y
            vert_aligned = abs(ci[0] - cj[0]) < ALIGN_TOL   # same x
            if horiz_aligned:
                dist = abs(ci[0] - cj[0])
                G.add_edge(i, j, dist=float(dist), align="horizontal")
            elif vert_aligned:
                dist = abs(ci[1] - cj[1])
                G.add_edge(i, j, dist=float(dist), align="vertical")

    # --- Visualise ---
    if image_stem is not None:
        _visualise(G, image_stem)

    return G


def _visualise(G: nx.Graph, image_stem: str):
    """Draw graph with nodes at their true centroid positions."""
    if len(G.nodes) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # match image coordinates (y=0 at top)
    ax.set_aspect("equal")
    ax.set_title(f"Spatial Graph: {image_stem}", fontsize=9)
    ax.axis("off")

    pos = {i: (d["centroid"][0], d["centroid"][1]) for i, d in G.nodes(data=True)}
    node_colors = [TYPE_COLORS.get(d.get("type", "unknown"), "#7f7f7f")
                   for _, d in G.nodes(data=True)]

    # Draw edges
    edge_styles = {"knn": ("grey", "--", 0.5), "horizontal": ("#888", "-", 0.8),
                   "vertical": ("#555", "-", 0.8)}
    for u, v, edata in G.edges(data=True):
        align = edata.get("align", "knn")
        color, ls, lw = edge_styles.get(align, ("grey", "--", 0.5))
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color=color, linestyle=ls, linewidth=lw, zorder=1)

    # Draw nodes
    node_ids = list(G.nodes)
    xs = [pos[i][0] for i in node_ids]
    ys = [pos[i][1] for i in node_ids]
    sizes = [max(20, G.nodes[i].get("area_ratio", 0.01) * 3000) for i in node_ids]
    ax.scatter(xs, ys, c=node_colors, s=sizes, zorder=2, edgecolors="white", linewidths=0.5)

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()
               if t in set(d.get("type", "unknown") for _, d in G.nodes(data=True))]
    ax.legend(handles=patches, loc="lower right", fontsize=7)

    out_path = GRAPHS_DIR / f"{image_stem}_graph.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    from layout_extraction import extract_layout
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python graph_builder.py <image_path>")
        sys.exit(1)
    from PIL import Image
    img = Image.open(path)
    shape = (img.height, img.width)
    elems = extract_layout(path)
    stem = Path(path).stem
    G = build_spatial_graph(elems, shape, image_stem=stem)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Graph saved to output/graphs/{stem}_graph.png")
