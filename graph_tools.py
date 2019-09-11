import json
import collections
from pprint import pprint

import networkx
from graphviz import Digraph

from circuit import Circuit, CircuitState, Op
from typing import Dict


def draw(circuit, feed_dict, previous_state):
    previous_state = collections.defaultdict(bool, previous_state)
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

def to_networkx(circuit: Circuit, feed_dict: CircuitState) -> networkx.DiGraph:
    net = networkx.DiGraph()
    sources = set(circuit.sources).union(feed_dict.keys())
    for op in circuit.ops:
        net.add_node(id(op),op=op.as_json,source=False,source_value=False)
    for source in sources:
        net.add_node(id(source),op={'op_type': 'source'},source=True, source_value=bool(feed_dict.get(source, source.default_value)))
        for op in circuit.downstream_ops(source):
            net.add_edge(id(source), id(op), line=source.as_json, id=id(source))
    for op in circuit.ops:
        for out_line in op.out_lines:
            for out_op in circuit.downstream_ops(out_line):
                net.add_edge(id(op), id(out_op), line=out_line.as_json, id=id(out_line))
    js = networkx.readwrite.json_graph.node_link_data(net)
    with open('graph.json', 'w') as fn:
        json.dump(js, fn)