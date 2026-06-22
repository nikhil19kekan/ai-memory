"""
Graph visualizer using networkx + matplotlib.
Renders after every update. New nodes/edges highlighted in the diff.
Reads from the new node_store + edge_store snapshot format.
"""

import networkx as nx
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Node color by entity type
NODE_COLORS = {
    "PERSON":  "#4A90D9",   # blue
    "PLACE":   "#E87B7B",   # red
    "ORG":     "#7BC6C0",   # teal
    "CONCEPT": "#C0C0C0",   # grey
    "UNKNOWN": "#BBBBBB",   # fallback grey
}

# Edge color by predicate
EDGE_COLORS = {
    "LIKES":     "#4A90D9",
    "LOVES":     "#1A3A8F",
    "HATES":     "#D94A4A",
    "DISLIKES":  "#E87B7B",
    "AVOIDS":    "#E8A838",
    "PREFERS":   "#7BC6C0",
    "COOKS":     "#E8A838",
    "EATS":      "#C07BC6",
    "WANTS":     "#4A90D9",
    "NEEDS":     "#4A90D9",
    "HAS":       "#7BC67E",
    "WORKS_AT":  "#7BC6C0",
    "LIVES_IN":  "#7BC6C0",
    "VISITS":    "#9B59B6",
    "KNOWS":     "#2ECC71",
    "COULD_BE":  "#7BC67E",
    "DEFAULT":   "#444444",
}


class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))

    def render(self, snapshot, highlight=None):
        """
        Build and draw the graph from the current snapshot.
        highlight: changes dict from GraphManager.process() — new items drawn bolder.
        Edge styles:
          - active + not inferred  → solid
          - active + inferred      → dotted
          - not active (superseded) → dashed (history)
        """
        G          = nx.DiGraph()
        nodes_data = snapshot.get("nodes", {})
        edges_data = snapshot.get("edges", {})

        # ── add nodes ────────────────────────────────────────────
        for name, nd in nodes_data.items():
            G.add_node(name, etype=nd["type"])

        # ── add edges ────────────────────────────────────────────
        for eid, ed in edges_data.items():
            subj = ed["subject"]
            obj  = ed["object"]
            # ensure endpoints exist even if missing from nodes (shouldn't happen)
            if subj not in G.nodes:
                G.add_node(subj, etype="UNKNOWN")
            if obj not in G.nodes:
                G.add_node(obj, etype="UNKNOWN")

            label = ed["predicate"]
            if ed.get("qualifier"):
                label += f"\n({ed['qualifier']})"
            label += f"\n{ed['date']}"
            if not ed.get("polarity", True):
                label = f"NOT {label}"

            G.add_edge(subj, obj,
                       label    = label,
                       etype    = ed["predicate"],
                       active   = ed.get("active",   True),
                       inferred = ed.get("inferred", False),
                       eid      = eid)

        # ── add attribute info nodes ─────────────────────────────
        for name, nd in nodes_data.items():
            for attr in nd.get("attributes", []):
                info_node = f"[{attr}]"
                G.add_node(info_node, etype="CONCEPT")
                G.add_edge(name, info_node, label="IS", etype="IS",
                           active=True, inferred=False, eid="")

        if len(G.nodes) == 0:
            return

        # ── layout ───────────────────────────────────────────────
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, k=2.5, seed=42)

        # ── collect highlighted new nodes/edges from diff ─────────
        new_node_names = set()
        new_edge_pairs = set()
        if highlight:
            for entry in highlight.get("new_entities", []):
                name = entry.split(" (")[0]
                new_node_names.add(name)
            for entry in highlight.get("new_edges", []):
                try:
                    u = entry.split(" --[")[0].strip()
                    v = entry.split("]--> ")[1].strip()
                    new_edge_pairs.add((u, v))
                except Exception:
                    pass

        # ── draw ─────────────────────────────────────────────────
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("Knowledge Graph", fontsize=14, fontweight="bold", pad=15)
        ax.axis("off")

        # node colors + sizes
        node_colors = []
        node_sizes  = []
        for n in G.nodes:
            etype = G.nodes[n].get("etype", "UNKNOWN")
            color = NODE_COLORS.get(etype, NODE_COLORS["UNKNOWN"])
            node_colors.append(color)
            node_sizes.append(2800 if n in new_node_names else 1800)

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes, alpha=0.92)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

        # split edges by active/inferred for different visual styles
        solid_edges    = []   # active, not inferred
        dotted_edges   = []   # active, inferred
        dashed_edges   = []   # superseded (history)
        highlighted    = []   # newly added (bold solid)

        for u, v, data in G.edges(data=True):
            if (u, v) in new_edge_pairs:
                highlighted.append((u, v))
            elif not data.get("active", True):
                dashed_edges.append((u, v))
            elif data.get("inferred", False):
                dotted_edges.append((u, v))
            else:
                solid_edges.append((u, v))

        def _colors(edge_list):
            return [
                EDGE_COLORS.get(G.edges[e].get("etype", "DEFAULT"), EDGE_COLORS["DEFAULT"])
                for e in edge_list
            ]

        draw_kwargs = dict(ax=ax, arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.12")

        if solid_edges:
            nx.draw_networkx_edges(G, pos, edgelist=solid_edges,
                                   edge_color=_colors(solid_edges),
                                   width=1.5, style="solid", **draw_kwargs)
        if dotted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=dotted_edges,
                                   edge_color=_colors(dotted_edges),
                                   width=1.5, style="dotted", **draw_kwargs)
        if dashed_edges:
            nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
                                   edge_color=_colors(dashed_edges),
                                   width=1.0, style="dashed", alpha=0.5, **draw_kwargs)
        if highlighted:
            nx.draw_networkx_edges(G, pos, edgelist=highlighted,
                                   edge_color=_colors(highlighted),
                                   width=3.5, style="solid", arrowsize=22,
                                   ax=ax, arrows=True, connectionstyle="arc3,rad=0.12")

        # edge labels (only on active edges to keep it readable)
        edge_labels = {
            (u, v): G.edges[u, v]["label"]
            for u, v in G.edges
            if G.edges[u, v].get("active", True)
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

        # legend — node types present
        present_types = {G.nodes[n].get("etype", "UNKNOWN") for n in G.nodes}
        legend = [
            mpatches.Patch(color=NODE_COLORS.get(t, "#BBBBBB"), label=t)
            for t in NODE_COLORS if t in present_types
        ]
        # edge style legend
        legend += [
            mpatches.Patch(color="#888888", label="── active"),
            mpatches.Patch(color="#888888", label="·· inferred"),
            mpatches.Patch(color="#cccccc", label="-- superseded"),
        ]
        if legend:
            ax.legend(handles=legend, loc="upper left", fontsize=8)

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.05)
