from graphviz import Digraph
import collections
from pprint import pprint

def key(x):
    return str(id(x))

def draw(circuit, feed_dict, previous_state):
    # pprint(circuit.lines.lines)
    d = Digraph()
    ops = list({
        op
        for line in previous_state
        for op in circuit.downstream_ops(line)
    })
    for op in ops:
        d.node(key(op))
    start_op = collections.defaultdict(bool, {
        line: op
        for op in ops
        for line in op.out_lines
    })
    end_ops = collections.defaultdict(set)
    for op in ops:
        for line in op.in_lines:
            end_ops[line].add(op)
    for line, value in previous_state.items():
        start_op_ = start_op[line]
        if not start_op_:
            d.node(key(line))
            start_op_ = line
        end_ops_ = end_ops[line]
        if not end_ops_:
            d.node(key(line))
            end_ops_ = {line}
        for end_op in end_ops_:
            d.edge(key(start_op_), key(end_op), f'{line.name[:10]} {1 if value else 0}')
    d.render('test-output/round-table.gv', view=True)

