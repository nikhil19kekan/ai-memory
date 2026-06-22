"""
CLI entry point.
Type sentences → graph updates + live visualization.
Type traversal queries → inspect the graph.
"""

from graph_manager import GraphManager
from visualizer import Visualizer
import traversal as tv


HELP = """
  Input sentences:
    <any sentence>            add to graph  e.g. "Nikhil likes biryani a lot"

  Traversal queries:
    about <entity>            everything known about an entity
    what does <entity> <pred> e.g. "what does nikhil like"
    who <pred> <entity>       e.g. "who likes pizza"
    timeline <entity>         full history including superseded facts
    compare <e1> <e2>         shared connections between two entities
    path <e1> <e2>            how are two entities connected

  Graph commands:
    dump                      print all nodes and active edges
    show                      redraw the graph
    quit / exit               exit
"""


def print_changes(changes):
    if changes["new_entities"]:
        print(f"  + entities : {', '.join(changes['new_entities'])}")
    for edge in changes["new_edges"]:
        print(f"  + edge     : {edge}")
    for attr in changes["attributes"]:
        print(f"  + info     : {attr}")
    for state in changes["states"]:
        print(f"  + state    : {state}")
    for inf in changes["inferred"]:
        print(f"  ~ inferred : {inf}")
    if changes["superseded"]:
        print(f"  ~ superseded: {', '.join(changes['superseded'])}")
    if changes["unknown_verbs"]:
        print(f"  ? unknown verbs (needs LLM): {', '.join(changes['unknown_verbs'])}")


def dump_graph(snapshot):
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    print(f"\n── nodes ({len(nodes)}) ──")
    for name, nd in sorted(nodes.items()):
        print(f"  {name!r}  [{nd['type']}]")
        if nd["attributes"]:
            for a in nd["attributes"]:
                print(f"    attr: {a}")
        active_states = [s for s in nd["states"] if s["active"]]
        if active_states:
            for s in active_states:
                print(f"    state: {s['state']} ({s['date']})")
        if nd["aliases"]:
            print(f"    aliases: {nd['aliases']}")

    active_edges = {eid: e for eid, e in edges.items() if e["active"]}
    print(f"\n── active edges ({len(active_edges)}) ──")
    for eid, e in sorted(active_edges.items()):
        label = f"  {e['subject']} --[{e['predicate']}"
        if e["qualifier"]:
            label += f", {e['qualifier']}"
        label += f" | {e['date']}]--> {e['object']}"
        if e["inferred"]:
            label += f"  (inferred, conf={e['confidence']:.1f})"
        print(label)

    inactive = {eid: e for eid, e in edges.items() if not e["active"]}
    if inactive:
        print(f"\n── superseded edges ({len(inactive)}) ──")
        for eid, e in sorted(inactive.items()):
            print(f"  [{eid}] {e['subject']} --[{e['predicate']} | {e['date']}]--> {e['object']}")


def handle_traversal(cmd, gm):
    """Parse and execute a traversal command. Returns True if handled."""
    words = cmd.strip().split()
    if not words:
        return False

    if words[0] == "about" and len(words) >= 2:
        entity = " ".join(words[1:])
        result = tv.about(gm.graph, entity)
        if "error" in result:
            print(f"  {result['error']}")
        else:
            print(f"\n  {result['entity']}  [{result['type']}]")
            if result["attributes"]:
                print(f"  attributes : {', '.join(result['attributes'])}")
            if result["states"]:
                for s in result["states"]:
                    print(f"  state      : {s['state']} ({s['date']})")
            if result["aliases"]:
                print(f"  aliases    : {', '.join(result['aliases'])}")
            if result["out_edges"]:
                print(f"  out edges:")
                for e in result["out_edges"]:
                    print(f"    {e}")
            if result["in_edges"]:
                print(f"  in edges:")
                for e in result["in_edges"]:
                    print(f"    {e}")
        return True

    if words[0] == "what" and len(words) >= 4 and words[1] == "does":
        entity    = words[2]
        predicate = words[3].upper()
        objects   = tv.what_does(gm.graph, entity, predicate)
        if objects:
            print(f"  {entity} {predicate.lower()}: {', '.join(objects)}")
        else:
            print(f"  no results for {entity} {predicate}")
        return True

    if words[0] == "who" and len(words) >= 3:
        predicate = words[1].upper()
        obj       = " ".join(words[2:])
        subjects  = tv.who_does(gm.graph, predicate, obj)
        if subjects:
            print(f"  who {predicate.lower()} {obj}: {', '.join(subjects)}")
        else:
            print(f"  no one found who {predicate.lower()} {obj}")
        return True

    if words[0] == "timeline" and len(words) >= 2:
        entity = " ".join(words[1:])
        edges  = tv.timeline(gm.graph, entity)
        if not edges:
            print(f"  no history for '{entity}'")
        else:
            print(f"\n  timeline for {entity}:")
            for e in edges:
                status = "SUPERSEDED" if not e.active else ("inferred" if e.inferred else "")
                tag    = f"  [{status}]" if status else ""
                print(f"    {e.date}  {e.subject} --[{e.predicate}]--> {e.object}{tag}")
        return True

    if words[0] == "compare" and len(words) >= 3:
        # compare e1 e2  (both entity names must be single words for simplicity)
        e1     = words[1]
        e2     = words[2]
        result = tv.compare(gm.graph, e1, e2)
        print(f"\n  comparing {e1} and {e2}:")
        if result["both_connect_to"]:
            print(f"  both connect to : {', '.join(result['both_connect_to'])}")
        else:
            print(f"  no shared connections")
        if result["connected_by"]:
            print(f"  both connected by: {', '.join(result['connected_by'])}")
        return True

    if words[0] == "path" and len(words) >= 3:
        e1    = words[1]
        e2    = words[2]
        found = tv.path(gm.graph, e1, e2)
        if not found:
            print(f"  no path found between {e1} and {e2}")
        else:
            print(f"  path: {tv.fmt_path(found)}")
        return True

    return False


def main():
    gm  = GraphManager()
    viz = Visualizer()

    print("Knowledge Graph  —  type 'help' for commands")
    print("─" * 50)

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not text:
            continue

        cmd = text.lower()

        if cmd in ("quit", "exit"):
            print("bye.")
            break

        if cmd == "help":
            print(HELP)
            continue

        if cmd == "dump":
            dump_graph(gm.snapshot())
            continue

        if cmd == "show":
            viz.render(gm.snapshot())
            continue

        # traversal commands
        if handle_traversal(cmd, gm):
            continue

        # ── process sentence ──────────────────────────────────
        changes = gm.process(text)
        print_changes(changes)
        viz.render(gm.snapshot(), highlight=changes)


if __name__ == "__main__":
    main()
