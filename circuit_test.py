from typing import Iterable

from pytest import fixture, skip
from pprint import pprint

import graph_tools

from circuit import (
    Line,
    zero_gate,
    alu,
    bus1,
    scope,
    Lines,
    lshift,
    rshift,
    and_gate,
    bit,
    bus,
    byte,
    decoder,
    inputs,
    nand,
    new_circuit,
    not_gate,
    or_gate,
    register,
    simulate,
    xor_gate,
    ram,
    adder,
    comp,
    comp_gate,
    cpu,
)


def test_line():
    assert Line("asdf").name == "asdf"
    assert len(Line()) == 1


def test_lines():
    a, b = Line("a"), Line("b")
    lines = Lines(a, b)
    assert list(lines) == [a, b]
    lines = Lines(b, a)
    assert list(lines) == [b, a]


def test_two_chain():
    a = Line("a")
    b = Line("b")
    assert list(a >> b) == [a, b]
    assert list(b >> a) == [b, a]


def test_three_chain():
    a, b, c = [Line(x) for x in "abc"]
    bc = b >> c
    assert list(bc >> a) == [b, c, a]
    assert list(a >> bc) == [a, b, c]
    assert list(a >> b >> c) == [a, b, c]
    assert list(c >> a >> b) == [c, a, b]


def test_lines_lines_chain():
    a, b, c, d = [Line(x) for x in "abcd"]
    assert list((a >> b) >> (c >> d)) == [a, b, c, d]
    assert list((a >> b) >> (c >> d) >> (b >> a >> d >> c)) == [a, b, c, d, b, a, d, c]


def test_lines_lookup(circuit):
    lines = Lines("abc")
    a, _, c = lines
    assert a.name == "a"
    assert lines["a"] == a
    assert lines["c"] == c
    assert lines["d"] is None


def test_line_eq():
    line_a1, line_a2 = Lines("aa")
    assert line_a1 == line_a1
    assert line_a2 == line_a2
    assert line_a1 != line_a2
    assert line_a2 != line_a1


def test_string_lines(circuit):
    assert "".join(line.name for line in Lines("abc")) == "abc"


@fixture
def circuit():
    with new_circuit() as circuit:
        yield circuit


def truth_table_test(lines: Lines, test_output, truth_table, draw=0):
    previous_state = None
    for i, (inputs, expectations) in enumerate(truth_table):
        if not isinstance(expectations, tuple):
            expectations = [expectations]
        n_digits = (
            len(lines)
            + 2
            - len([value for value in inputs if not isinstance(value, str)])
        )
        inputs = [
            v
            for value in inputs
            for v in (
                [value]
                if isinstance(value, int)
                else map(int, format(int(value), f"#0{n_digits}b")[2:])
            )
        ]
        n_digits = (
            len(test_output)
            + 2
            - len([value for value in expectations if not isinstance(value, str)])
        )
        expectations = [
            v
            for value in expectations
            for v in (
                [value]
                if isinstance(value, int)
                else map(int, format(int(value), f"#0{n_digits}b")[2:])
            )
        ]
        feed_dict = dict(zip(lines, inputs))
        previous_state = simulate(feed_dict, previous_state=previous_state, draw=draw)
        print(
            i,
            expectations,
            [int(previous_state[output_line]) for output_line in test_output],
        )
        for output_line, expectation in zip(test_output, expectations):
            assert previous_state[output_line] == expectation


def test_nand(circuit):
    a, b = inputs("ab")
    truth_table_test(
        [a, b], a >> b >> nand, (((1, 1), 0), ((1, 0), 1), ((0, 1), 1), ((0, 0), 1))
    )


def test_and(circuit):
    a, b = inputs("ab")
    truth_table_test(
        [a, b], a >> b >> and_gate, (((1, 1), 1), ((1, 0), 0), ((0, 1), 0), ((0, 0), 0))
    )


def test_or(circuit):
    a, b = inputs("ab")
    truth_table_test(
        [a, b], a >> b >> or_gate, (((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0))
    )


def test_xor(circuit):
    a, b = inputs("ab")
    truth_table_test(
        [a, b], a >> b >> xor_gate, (((1, 1), 0), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0))
    )


def test_not(circuit):
    a, b = inputs("ab")
    truth_table_test(
        [a, b],
        a >> b >> not_gate,
        (((1, 1), (0, 0)), ((1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, 0), (1, 1))),
    )


def test_hash():
    l = Line("a")
    h = hash(l)
    l.set_name("asdf")
    assert l.name == "asdf"
    h2 = hash(l)
    assert h == h2


def test_bit(circuit):
    input_lines = inputs("is")
    truth_table_test(
        input_lines,
        input_lines >> bit,
        (
            ((1, 1), 1),
            ((0, 1), 0),
            ((0, 0), 0),
            ((1, 0), 0),
            ((0, 0), 0),
            ((0, 1), 0),
            ((1, 1), 1),
            ((1, 0), 1),
            ((0, 0), 1),
            ((1, 0), 1),
            ((1, 1), 1),
            ((1, 0), 1),
            ((0, 1), 0),
            ((1, 0), 0),
        ),
    )


def test_bus():
    assert len(bus()) == 8


def test_byte(circuit):
    inputs = bus() >> Line("s")
    outputs = inputs >> byte
    assert len(inputs) == 9
    assert len(outputs) == 8
    truth_table_test(
        inputs,
        outputs,
        (
            (("1", 1), "1"),
            (("25", 1), "25"),
            (("25", 0), "25"),
            (("42", 0), "25"),
            (("42", 1), "42"),
            (("0", 1), "0"),
            (("251", 1), "251"),
            (("120", 1), "120"),
        ),
    )


def test_register(circuit):
    inputs = bus() >> Lines("se")
    outputs = inputs >> register
    truth_table_test(
        inputs,
        outputs,
        (
            (("1", 1, 1), "1"),
            (("1", 1, 0), "0"),
            (("1", 0, 0), "0"),
            (("1", 0, 1), "1"),
            (("25", 0, 1), "1"),
            (("25", 0, 0), "0"),
            (("25", 0, 1), "1"),
            (("25", 1, 0), "0"),
            (("4", 0, 1), "25"),
            (("4", 1, 0), "0"),
            (("4", 0, 0), "0"),
            (("12", 0, 1), "4"),
            (("12", 1, 1), "12"),
        ),
    )


def test_decoder(circuit):
    n_inputs = 3
    inputs = bus(n_inputs)
    outputs = inputs >> decoder
    assert len(outputs) == 2 ** n_inputs
    truth_table_test(
        inputs, outputs, (((str(i),), str(2 ** i)) for i in range(2 ** n_inputs))
    )


def test_ram(circuit):
    with scope("BusInput"):
        inputs = bus()
        s = Line("s")
        e = Line("e")
    with scope("MarInput"):
        mar_inputs = bus()
        sa = Line("sa")
    previous_state = {}
    all_inputs = s >> e >> inputs >> sa >> mar_inputs
    all_inputs >> ram
    mar_inputs = list(mar_inputs)
    inputs = list(inputs)
    mi1 = mar_inputs[0]
    i1 = inputs[0]
    i2 = inputs[1]
    feed_dict = {}

    def run():
        previous_state.update(
            simulate(feed_dict, circuit, previous_state=previous_state)
        )

    def setv(line, value):
        feed_dict[line] = value

    def chkv(line, value):
        assert previous_state[line] == value

    def rmv(line):
        del feed_dict[line]

    def viz(skip=False):
        graph_tools.draw(circuit, feed_dict, previous_state)
        # skip()

    setv(s, 1)
    setv(sa, 1)
    run()
    setv(mi1, 1)
    run()
    setv(sa, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(i1, 1)
    run()
    chkv(i1, 1)
    setv(s, 1)
    run()
    chkv(i1, 1)
    setv(s, 0)
    run()
    chkv(i1, 1)
    rmv(i1)
    run()
    chkv(i1, 0)
    setv(e, 1)
    run()
    chkv(i1, 1)
    chkv(i2, 0)
    setv(e, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(e, 1)
    run()
    chkv(i1, 1)
    chkv(i2, 0)
    setv(e, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(mi1, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(sa, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(sa, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(e, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(e, 0)
    run()
    setv(i1, 0)
    setv(i2, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 1)
    setv(s, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 1)
    setv(s, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 1)
    rmv(i1)
    rmv(i2)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(e, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 1)
    setv(e, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(e, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 1)
    setv(e, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    setv(mi1, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    chkv(mi1, 1)
    setv(sa, 1)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    chkv(mi1, 1)
    setv(sa, 0)
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    chkv(mi1, 1)
    setv(e, 1)
    run()
    chkv(i1, 1)
    chkv(i2, 0)
    setv(mi1, 0)
    run()
    chkv(i1, 1)
    chkv(i2, 0)
    viz()


def test_adder(circuit):
    with scope("a"):
        a_values = bus(2)
    with scope("b"):
        b_values = bus(2)
    carry_in = Line("CarryIn")
    inputs = carry_in >> a_values >> b_values
    output = inputs >> adder
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0, 0, 0), (0, 0, 0)),
            ((1, 0, 0, 0, 0), (0, 1, 0)),
            ((0, 1, 0, 0, 0), (0, 1, 0)),
            ((0, 0, 0, 1, 0), (0, 1, 0)),
            ((0, 1, 0, 1, 0), (0, 0, 1)),
            ((1, 1, 0, 1, 0), (0, 1, 1)),
            ((1, 1, 1, 1, 1), (1, 1, 1)),
            ((0, 1, 1, 1, 1), (1, 0, 1)),
            ((0, 1, 0, 1, 1), (1, 0, 0)),
            ((0, 1, 1, 1, 0), (1, 0, 0)),
            ((0, 0, 1, 0, 0), (0, 0, 1)),
            ((0, 0, 0, 0, 1), (0, 0, 1)),
            ((0, 0, 0, 1, 1), (0, 1, 1)),
            ((0, 1, 1, 0, 0), (0, 1, 1)),
        ),
        draw=1,
    )


def test_comparator(circuit):
    with scope("a"):
        a_values = bus(2)
    with scope("b"):
        b_values = bus(2)
    inputs = a_values >> b_values
    output = inputs >> comp
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0, 0), (1, 0, 0, 0)),
            ((0, 0, 1, 0), (0, 0, 1, 0)),
            ((0, 0, 0, 1), (0, 0, 0, 1)),
            ((0, 0, 1, 1), (0, 0, 1, 1)),
            ((1, 0, 0, 0), (0, 1, 1, 0)),
            ((1, 0, 1, 0), (1, 0, 0, 0)),
            ((1, 0, 0, 1), (0, 1, 1, 1)),
            ((1, 0, 1, 1), (0, 0, 0, 1)),
            ((0, 1, 0, 0), (0, 1, 0, 1)),
            ((0, 1, 1, 0), (0, 0, 1, 1)),
            ((0, 1, 0, 1), (1, 0, 0, 0)),
            ((0, 1, 1, 1), (0, 0, 1, 0)),
            ((1, 1, 0, 0), (0, 1, 1, 1)),
            ((1, 1, 1, 0), (0, 1, 0, 1)),
            ((1, 1, 0, 1), (0, 1, 1, 0)),
            ((1, 1, 1, 1), (1, 0, 0, 0)),
        ),
        draw=1,
    )


def test_comparator_gate(circuit):
    inputs = Lines("abel")
    output = inputs >> comp_gate
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0, 0), (0, 0, 0)),
            ((0, 0, 1, 0), (1, 0, 0)),
            ((0, 0, 0, 1), (0, 1, 0)),
            ((0, 0, 1, 1), (1, 1, 0)),
            ((1, 0, 0, 0), (0, 0, 1)),
            ((1, 0, 1, 0), (0, 1, 1)),
            ((1, 0, 0, 1), (0, 1, 1)),
            ((1, 0, 1, 1), (0, 1, 1)),
            ((0, 1, 0, 0), (0, 0, 1)),
            ((0, 1, 1, 0), (0, 0, 1)),
            ((0, 1, 0, 1), (0, 1, 1)),
            ((0, 1, 1, 1), (0, 1, 1)),
            ((1, 1, 0, 0), (0, 0, 0)),
            ((1, 1, 1, 0), (1, 0, 0)),
            ((1, 1, 0, 1), (0, 1, 0)),
            ((1, 1, 1, 1), (1, 1, 0)),
        ),
        draw=1,
    )


def test_zero_gate(circuit):
    inputs = bus(3)
    output = inputs >> zero_gate
    truth_table_test(
        inputs,
        output,
        (((0, 0, 0), 1), ((1, 0, 0), 0), ((0, 1, 1), 0), ((1, 1, 1), 0)),
        draw=1,
    )


def test_lshifter(circuit):
    inputs = bus(3)
    output = inputs >> lshift
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (0, 1, 0)),
            ((0, 1, 0), (1, 0, 0)),
            ((0, 1, 1), (1, 1, 0)),
            ((1, 0, 0), (0, 0, 1)),
            ((1, 0, 1), (0, 1, 1)),
            ((1, 1, 0), (1, 0, 1)),
            ((1, 1, 1), (1, 1, 1)),
        ),
        draw=1,
    )


def test_rshifter(circuit):
    inputs = bus(3)
    output = inputs >> rshift
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (0, 0, 1)),
            ((0, 1, 1), (1, 0, 1)),
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 1), (1, 1, 0)),
            ((1, 1, 0), (0, 1, 1)),
            ((1, 1, 1), (1, 1, 1)),
        ),
        draw=1,
    )


def test_alu(circuit):
    with scope("a"):
        a = bus(4)
    with scope("b"):
        b = bus(4)
    with scope("op"):
        op = bus(3)
    carry_in = Line("CarryIn")
    inputs = op >> carry_in >> a >> b
    a_larger, equal, out_zero, carry_out, cs = (inputs >> alu).split(1,1,1,1)
    feed_dict = {i: 0 for i in inputs}
    feed_dict[op[0]] = 0
    feed_dict[op[1]] = 0
    feed_dict[op[2]] = 0
    feed_dict[a[0]] = 1
    feed_dict[b[1]] = 1
    feed_dict[a[3]] = 1
    feed_dict[b[3]] = 1
    simulation = simulate(feed_dict, circuit)
    graph_tools.draw(circuit, feed_dict, simulation)
    assert simulation[cs[0]]
    assert simulation[cs[1]]
    assert not simulation[cs[2]]
    assert not simulation[cs[3]]
    assert simulation[carry_out]
    assert not simulation[out_zero]
    
    feed_dict = {i: 0 for i in inputs}
    feed_dict[op[0]] = 1
    feed_dict[op[1]] = 1
    feed_dict[op[2]] = 0
    feed_dict[a[1]] = 1
    feed_dict[b[2]] = 1
    feed_dict[a[3]] = 1
    feed_dict[b[3]] = 1
    simulation = simulate(feed_dict, circuit)
    graph_tools.draw(circuit, feed_dict, simulation)
    assert not simulation[cs[0]]
    assert simulation[cs[1]]
    assert simulation[cs[2]]
    assert not simulation[cs[3]]
    assert not simulation[carry_out]
    assert not simulation[out_zero]
    assert simulation[a_larger]
    assert not simulation[equal]

    feed_dict = {i: 0 for i in inputs}
    feed_dict[op[0]] = 0
    feed_dict[op[1]] = 0
    feed_dict[op[2]] = 1
    feed_dict[a[2]] = 1
    feed_dict[b[2]] = 1
    feed_dict[a[3]] = 1
    feed_dict[b[3]] = 1
    feed_dict[carry_in] = 1
    simulation = simulate(feed_dict, circuit)
    graph_tools.draw(circuit, feed_dict, simulation)
    assert simulation[cs[0]]
    assert not simulation[cs[1]]
    assert not simulation[cs[2]]
    assert simulation[cs[3]]
    assert simulation[carry_out]
    assert not simulation[out_zero]

    feed_dict = {i: 0 for i in inputs}
    feed_dict[op[0]] = 0
    feed_dict[op[1]] = 0
    feed_dict[op[2]] = 1
    feed_dict[a[2]] = 1
    feed_dict[b[2]] = 1
    feed_dict[a[3]] = 1
    feed_dict[b[3]] = 1
    feed_dict[carry_in] = 0
    simulation = simulate(feed_dict, circuit)
    # graph_tools.draw(circuit, feed_dict, simulation)
    assert not simulation[cs[0]]
    assert not simulation[cs[1]]
    assert not simulation[cs[2]]
    assert simulation[cs[3]]
    assert simulation[carry_out]
    assert not simulation[out_zero]
    
def test_bus1(circuit):
    inputs = bus(3)
    s = Line('s')
    inputs = s >> inputs
    output = inputs >> bus1
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0, 0), (0, 0, 0)),
            ((0, 1, 1, 0), (1, 1, 0)),
            ((0, 1, 0, 1), (1, 0, 1)),
            ((1, 1, 1, 0), (0, 0, 1)),
            ((1, 0, 0, 0), (0, 0, 1)),
            ((1, 0, 0, 1), (0, 0, 1)),
        ),
        draw=1,
    )

def test_cpu(circuit):
    with scope("Signals"):
        s,e, carry_in = Lines('se') >> Line("CarryIn")
    with scope('Op'):
        op = bus(3)
    with scope("Bus"):
        b = bus()
    inputs = s >> e >> op >> carry_in >> b
    output = inputs >> cpu
    feed_dict = {i: 0 for i in inputs}
    feed_dict[s] = 1
    feed_dict[e] = 1
    feed_dict[op[1]] = 1
    feed_dict[b[0]] = 1
    feed_dict[b[2]] = 1
    simulation = simulate(feed_dict, circuit)
    graph_tools.draw(circuit, feed_dict, simulation)