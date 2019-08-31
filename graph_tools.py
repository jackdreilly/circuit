from graphviz import Digraph
import collections
from pprint import pprint


def draw(circuit, feed_dict, previous_state):
    ops = list({op for line in previous_state for op in circuit.downstream_ops(line)})
    nodes = []
    edges = []
    groups = set()
    for op in ops:
        group = op.name[: op.name.rfind("/")]
        groups.add(group)
        nodes.append({"key": str(id(op)), "name": op.name, "group": group})
    for line, value in previous_state.items():
        name = line.name
        ind = name.rfind("/")
        group = ""
        if ind > 0:
            group = name[:ind]
            name = name[ind+1:]
        groups.add(group)
        nodes.append({"key": str(id(line)), "name": name, "group": group})
    for op in ops:
        for line in op.out_lines:
            value = previous_state[line]
            edges.append(
                {
                    "from": str(id(op)),
                    "to": str(id(line)),
                    "name": f"{1 if value else 0} {line.name}",
                    "color": "#f00" if value else '#000',
                }
            )
        for line in op.in_lines:
            value = previous_state[line]
            edges.append(
                {
                    "from": str(id(line)),
                    "to": str(id(op)),
                    "name": f"{1 if value else 0} {line.name}",
                    "color": "#f00" if value else '#000',
                    "thickness": 2 if value else 1,
                }
            )
    queue = collections.deque(groups)
    visited = set()
    while queue:
        group = queue.pop()
        if group in visited:
            continue
        visited.add(group)
        if "/" not in group:
            nodes.append({"key": group, "isGroup": True, "name": group or "Inputs"})
            continue
        ind = group.rfind("/")
        group_group = group[:ind]
        name = group[ind + 1 :]
        nodes.append(
            {
                "key": group,
                "name": name or "Inputs",
                "isGroup": True,
                "group": group_group,
            }
        )
        queue.append(group_group)
    with open("data.json", "w") as fn:
        import json
        json.dump({"linkDataArray": edges, "nodeDataArray": nodes}, fn)

