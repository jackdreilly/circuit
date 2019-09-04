import parser
from pprint import pprint
from typing import Any, Iterable, List
import pytest
from pytest import fixture, skip

import circuit as circuit_module
import graph_tools
from circuit import (Line, Lines, adder, alu, and_gate, bit, bus, bus1, byte,
                     comp, comp_gate, cpu, decoder, inputs, lshift, nand,
                     new_circuit, not_gate, or_gate, ram, register, rshift,
                     scope, simulate, stepper, xor_gate, zero_gate)


def test_line(circuit):
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


@fixture
def small_circuit():
    old = ram.MEM_SIZE
    ram.MEM_SIZE = 4
    with new_circuit() as circuit:
        yield circuit
    ram.MEM_SIZE = old

@fixture
def medium_circuit():
    old = ram.MEM_SIZE
    ram.MEM_SIZE = 6
    with new_circuit() as circuit:
        yield circuit
    ram.MEM_SIZE = old


@fixture
def smallest_circuit():
    old = ram.MEM_SIZE
    ram.MEM_SIZE = 2
    with new_circuit() as circuit:
        yield circuit
    ram.MEM_SIZE = old


def truth_table_test(lines: Lines, test_output, truth_table, draw=0):
    previous_state = None
    for i, (inputs, expectations) in enumerate(truth_table):
        if not isinstance(expectations, tuple) and not isinstance(expectations, list):
            expectations = [expectations]
        inputs = [
            v
            for value in inputs
            for v in (
                [value]
                if isinstance(value, int)
                else None
                if value is None
                else [None] * circuit_module.BYTE_SIZE
                if value == ""
                else map(int, format(int(value), f"#010b")[2:])
            )
        ]
        expectations = [
            v
            for value in expectations
            for v in (
                [value]
                if isinstance(value, int)
                else None
                if value is None
                else [None] * circuit_module.BYTE_SIZE
                if value == ""
                else map(int, format(int(value), f"#010b")[2:])
            )
        ]
        feed_dict = dict(zip(lines, inputs))
        feed_dict = {k: v for k, v in feed_dict.items() if v is not None}
        pprint(feed_dict)
        pprint(previous_state)
        previous_state = simulate(feed_dict, previous_state=previous_state)
        if draw and False:
            graph_tools.draw(circuit_module.default_circuit(), feed_dict, previous_state)
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
            ((0, 0), 1),
            ((0, 1), []),
            ((0, 1), 0),
            ((1, 0), 0),
        ),
    )


def test_bus():
    assert len(bus()) == 8


def test_byte(circuit):
    inputs = Line("s") >> bus()
    outputs = inputs >> byte
    assert len(inputs) == 9
    assert len(outputs) == 8
    truth_table_test(
        inputs,
        outputs,
        (
            ((1, "1"), "1"),
            ((1, "25"), "25"),
            ((0, "25"), "25"),
            ((0, "42"), "25"),
            ((1, "42"), []),
            ((1, "42"), "42"),
            ((1, "0"), "0"),
            ((1, "251"), "251"),
            ((1, "120"), "120"),
        ),
        draw=1,
    )


def test_register(circuit):
    inputs = Lines("se") >> bus()
    outputs = inputs >> register
    truth_table_test(
        inputs,
        outputs,
        (
            ((1, 1, "1"), "1"),
            ((1, 0, "1"), "0"),
            ((0, 0, "1"), "0"),
            ((0, 1, "1"), "1"),
            ((0, 1, "25"), "1"),
            ((0, 0, "25"), "0"),
            ((0, 1, "25"), "1"),
            ((1, 0, "25"), "0"),
            ((0, 1, "4"), "25"),
            ((1, 0, "4"), "0"),
            ((0, 0, "4"), "0"),
            ((0, 1, "12"), "4"),
            ((1, 1, "12"), "12"),
        ),
    )


def test_decoder(circuit):
    n_inputs = 4
    inputs = bus(n_inputs)
    outputs = inputs >> decoder
    assert len(outputs) == 2 ** n_inputs
    truth_table_test(
        inputs,
        outputs,
        [
            (
                list(map(int, "{0:06b}".format(i)[2:])),
                [int(i == j) for j in range(2 ** n_inputs)][::-1],
            )
            for i in range(2 ** n_inputs)
        ],
    )


def test_ram(small_circuit):
    circuit = small_circuit
    with scope("BusInput"):
        inputs = bus()
        s = Line("s")
        e = Line("e")
    with scope("MarInput"):
        mar_inputs = bus()
        sa = Line("sa")
    previous_state = {}
    all_inputs = s >> e >> sa >> inputs >> mar_inputs
    all_inputs >> ram
    mi1 = mar_inputs[-1]
    i1 = inputs[-1]
    i2 = inputs[-2]
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
    
    # S set
    setv(s, 1)
    # MAR set
    setv(sa, 1)
    run()
    # MAR 1 set
    setv(mi1, 1)
    run()
    # MAR unset
    setv(sa, 0)
    run()
    # INPUT 0 0
    chkv(i1, 0)
    chkv(i2, 0)
    # CHANGE INPUT to 1 0
    setv(i1, 1)
    run()
    # CHECK INPUT STILL ACTIVE
    chkv(i1, 1)
    chkv(i2, 0)
    setv(s, 1)
    # CHECK INPUT STILL ACTIVE
    run()
    # Remove S
    chkv(i1, 1)
    setv(s, 0)
    run()
    # In still exists
    chkv(i1, 1)
    rmv(i1)
    run()
    # E disabled
    chkv(i1, 0)
    setv(e, 1)
    run()
    # E Enabled, I persists from bytes
    chkv(i1, 1)
    chkv(i2, 0)
    setv(e, 0)
    # Disable E
    run()
    # Values turn off again
    chkv(i1, 0)
    chkv(i2, 0)
    # Reenable E
    setv(e, 1)
    run()
    # Values come back
    chkv(i1, 1)
    chkv(i2, 0)
    # Disable again
    setv(e, 0)
    run()
    # disabled
    chkv(i1, 0)
    chkv(i2, 0)
    # MI Changed to 0
    setv(mi1, 0)
    # Nothing enabled
    run()
    chkv(i1, 0)
    chkv(i2, 0)
    # Set MAR turned on
    setv(sa, 1)
    run()
    # MAR is now set to 0
    chkv(i1, 0)
    chkv(i2, 0)
    setv(sa, 0)
    run()
    # MAR disabled
    chkv(i1, 0)
    chkv(i2, 0)
    # E re-enabled
    setv(e, 1)
    run()
    viz()
    # But no input value has been set
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
            ((1, 0, 0, 0, 0), (0, 0, 1)),
            ((0, 1, 0, 0, 0), (0, 1, 0)),
            ((0, 0, 0, 1, 0), (0, 1, 0)),
            ((0, 1, 0, 1, 0), (1, 0, 0)),
            ((1, 1, 0, 1, 0), (1, 0, 1)),
            ((1, 1, 1, 1, 1), (1, 1, 1)),
            ((0, 1, 1, 1, 1), (1, 1, 0)),
            ((0, 1, 0, 1, 1), (1, 0, 1)),
            ((0, 1, 1, 1, 0), (1, 0, 1)),
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
    a_larger, equal, out_zero, carry_out, cs = (inputs >> alu).split(1, 1, 1, 1)
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
    assert simulation[cs[2]]
    assert not simulation[cs[3]]
    assert not simulation[carry_out]
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
    s = Line("s")
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

def run_cpu(medium_circuit):
    circuit = medium_circuit
    clock = circuit_module.Clock()
    output = clock.lines >> cpu
    iars = tag_outputs(circuit, ["BIT"], "Cpu/IAR")
    irs = tag_outputs(circuit, ["BIT"], "Cpu/IR")
    n_rows = 2 ** (ram.MEM_SIZE // 2)
    n_cols = 2 ** ((ram.MEM_SIZE + 1) // 2)
    rams = {(row,col): tag_outputs(circuit, ["BIT"], f'reg-{row}-{col}') for row in range(n_rows) for col in range(n_cols)}
    regs = {i: tag_outputs(circuit, ["BIT"], f"Cpu/Registers/{i}") for i in range(cpu.N_REGISTERS)}
    bus = tag_outputs(circuit, ["MAINBUS"])
    stepper = tag_outputs(circuit, ["STEPPER"])
    selectors = tag_outputs(circuit, ["SELECTOR", "CONTROL"])
    enablers = tag_outputs(circuit, ["ENABLER", "CONTROL"])
    mars = tag_outputs(circuit, ["MAROUTPUT"])
    acc = tag_outputs(circuit, ["BIT"],"ACC" )
    flags = tag_outputs(circuit, ["FLAG"])
    print(flags)
    assert len(flags) == 4
    bootloader_program = circuit_module.bootloader_program()
    program_line = bootloader_program[0]
    simulation = {}
    def set_lines(lines, vals, prev=None):
        if prev is None:
            prev = {}
        prev.update({l: v for l,v in zip(lines, vals)})
        return prev

    bl_length = 6

    my_program = parser.parse("""
    XOR  0 0;
    DATA 1 1;
    DATA 2 2;
    ADD  0 2;
    XOR  0 0;
    ADD  1 0;
    XOR  1 1;
    ADD  2 1;
    OUT  1  ;
    JMP  19 ;
    """)
    print("MY PROGRAM")
    pprint(my_program)
    print("BL PROGRAM")
    pprint(bootloader_program)
    
    
    def bootload(program):
        program_iter = iter(program)
        d = {}
        for row in range(n_rows):
            for col in range(n_cols):
                try:
                    d = set_lines(tag_outputs(circuit, ["BIT"], f'reg-{row}-{col}'), next(program_iter), d)
                except StopIteration:
                    return d

    fixed_dict = bootload(bootloader_program)
    def vals(lines, keys=False,show_all=False,decimal=False):
        if show_all:
            return [(line.name, int(simulation[line])) for line in lines]
        if keys:
            return [(line.name, int(simulation[line])) for line in lines if simulation[line]]
        if decimal:
            return int(''.join(map(str,[int(simulation[line]) for line in lines])), 2)
        return '{0:02x}'.format(int(''.join(map(str,[int(simulation[line]) for line in lines])), 2))
    io_lines = tag_outputs(circuit, ("IO", ("IO", "IN")))
    out_lines = tag_outputs(circuit, ("IO", ("IO", "OUT")))
    def input_dict(step):
        step-=3
        if step < 0:
            return {}
        step //= bl_length
        if step >= len(my_program):
            return {}
        return {line: v for line, v in zip(io_lines, my_program[step])}
    for i in range(24 * 1000):
        feed_dict = clock.step()
        feed_dict.update(fixed_dict)
        feed_dict.update(input_dict(i // 24))
        simulation.update(simulate(feed_dict, circuit, simulation))
        if (not (i + 1) % 24):
            print('CYC', i % 4, 'RND', i // 24, "STEP", (i % 24) // 4)
            # print("CLOCK", vals(clock.lines))
            print("IRS", vals(irs), parser.unparse(vals(irs, decimal=True)))
            print("BUS", vals(bus))
            print("IAR", vals(iars))
            print("MARS", vals(mars))
            print("RAMS")
            print('\n'.join(' '.join(vals(rams[(row,col)]) for col in range(n_cols)) for row in range(n_rows)))
            print("REGS")
            print(' '.join(vals(regs[i]) for i in range(len(regs))))
            print("FLAGS", vals(flags,keys=True))
            print("STEPPER", vals(stepper))
            print("ACC", vals(acc))
            print("ENABLERS", vals(enablers,keys=True))
            print("SELECTORS", vals(selectors, keys=True))
            print("IO-IN", vals(io_lines), parser.unparse(vals(io_lines, decimal=True)))
            print("IO-OU", vals(out_lines))
    graph_tools.draw(circuit, feed_dict, simulation)

def test_stepper_2(circuit):
    clock = circuit_module.Clock()
    output = clock.clk >> stepper
    simulation = {}
    expected = [0] * 6
    for step in range(6):
        for _ in range(4):
            expected = [0] * 6
            expected[step] = 1
            simulation.update(
                simulate({clock.clk: clock.step()[clock.clk]}, circuit, simulation)
            )
            assert [simulation[line] for line in output] == expected


def test_clock(circuit):
    clock = circuit_module.Clock()
    simulation = {}

    def check(*v):
        feed_dict = clock.step()
        simulation.update(simulate(feed_dict, circuit, simulation))
        graph_tools.draw(circuit, feed_dict, simulation)
        return v == [
            simulation[l]
            for l in [clock.clk, clock.delay_clk, clock.clk_s, clock.clk_e]
        ]

    check(1, 1, 1, 1)
    check(0, 1, 0, 1)
    check(0, 0, 0, 0)
    check(1, 0, 0, 1)
    check(1, 1, 1, 1)
    check(0, 1, 0, 1)
    check(0, 0, 0, 0)
    check(1, 0, 0, 1)
    check(1, 1, 1, 1)
    check(0, 1, 0, 1)
    check(0, 0, 0, 0)
    check(1, 0, 0, 1)


def tag_outputs(circuit, tags, name_filter=""):
    return Lines(
        l
        for l in sorted(set(circuit.lines.typed(*tags)), key=lambda x: x.name)
        if name_filter in l.name
    )


def print_lines(lines, sim):
    pprint([((l.name), int(sim[l])) for l in lines])


def run_controller(circuit, instruction: str, flags=""):
    parsed, _ = parser.parse_line(instruction)
    print(instruction, parsed)
    clock = circuit_module.Clock()
    with scope("IR"):
        ir = bus()
    with scope("FLAGS"):
        flags_in = Lines("CAEZ")
    with scope("STEPPER"):
        stepper = bus(6)
    inputs = clock.clk >> clock.clk_s >> clock.clk_e >> ir >> flags_in >> stepper
    output = inputs >> circuit_module.controller
    simulation = {}
    es = tag_outputs(circuit, ["CONTROL", "ENABLER"])
    ss = tag_outputs(circuit, ["CONTROL", "SELECTOR"])
    rounds = []
    for step in range(6):
        e_found = set()
        s_found = set()
        for clock_round in range(4):
            f = clock.step()
            f.update({stepper[j]: int(j == step) for j in range(6)})
            f.update({ir[j]: k for j, k in enumerate(parsed)})
            f.update(
                {flags_in[j]: int(letter in flags) for j, letter in enumerate("CAEZ")}
            )
            simulation.update(simulate(f, circuit, simulation))
            new_es = set(e for e in es if simulation[e])
            new_ss = set(s for s in ss if simulation[s])
            if clock_round == 3:
                assert (
                    len(
                        [
                            line
                            for line in new_es
                            if line not in Lines(new_es).typed(("E", "B1"))
                            and line not in Lines(new_es).typed(("ALU", "OP"))
                        ]
                    )
                    == 0
                )
            if clock_round != 1:
                assert len(new_ss) == 0
            e_found.update(new_es)
            s_found.update(new_ss)
        rounds.append({"e": Lines(e_found), "s": Lines(s_found)})
    return rounds


def is_types(lines: Lines, types: List[List[Any]]):
    assert len(lines) == len(types), (lines, types)
    for line_type in types:
        assert len(lines.typed(*line_type)) == 1, (lines, line_type)


def assert_round(round_, assertion):
    for key, key_assert in zip(["e", "s"], assertion):
        is_types(round_[key], key_assert)


def assert_rounds(rounds, assertions):
    for round_, assertion in zip(rounds, assertions):
        if assertion is None:
            continue
        assert_round(round_, assertion)


def assert_command(circuit, command, assertions, **kwargs):
    assert_rounds(run_controller(circuit, command, **kwargs), assertions)


def assert_command_with_fetch(circuit, command, assertions, **kwargs):
    assert_command(circuit, command, fetch_assertions() + assertions, **kwargs)


def fetch_assertions():
    return [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "RAM")]], [[("S", "IR")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
    ]


def test_load_controller(circuit):
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "MAR")]]),
        ([[("E", "RAM")]], [[("S", "R"), ("R", 2)]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "LD 1 2", assertions)
    assertions = [
        ([[("E", "R"), ("R", 0)]], [[("S", "MAR")]]),
        ([[("E", "RAM")]], [[("S", "R"), ("R", 3)]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "LD 0 3", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "MAR")]]),
        ([[("E", "RAM")]], [[("S", "R"), ("R", 2)]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "LD 2 2", assertions)


def test_store_controller(circuit):
    assertions = [
        ([[("E", "R"), ("R", 3)]], [[("S", "MAR")]]),
        ([[("E", "R"), ("R", 0)]], [[("S", "RAM")]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "ST 3 0", assertions)
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "MAR")]]),
        ([[("E", "R"), ("R", 1)]], [[("S", "RAM")]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "ST 1 1", assertions)


def test_jmpr_controller(circuit):
    assertions = [([[("E", "R"), ("R", 3)]], [[("S", "IAR")]]), ([], []), ([], [])]
    assert_command_with_fetch(circuit, "JMPR 3", assertions)
    assertions = [([[("E", "R"), ("R", 1)]], [[("S", "IAR")]]), ([], []), ([], [])]
    assert_command_with_fetch(circuit, "JMPR 1", assertions)


def test_jmp_controller(circuit):
    assertions = [
        ([[("E", "IAR")]], [[("S", "MAR")]]),
        ([[("E", "RAM")]], [[("S", "IAR")]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "JMP 24", assertions)


def test_clf_controller(circuit):
    assertions = [([[("E", "B1")]], [[("S", "FLAGS")]]), ([], []), ([], [])]
    assert_command_with_fetch(circuit, "CLF", assertions)


def test_io_controller(circuit):
    assertions = [([], []), ([[("E", "IO")]], [[("S", "R"), ("R", 3)]]), ([], [])]
    assert_command_with_fetch(circuit, "IN 3", assertions)
    assertions = [([], []), ([[("E", "IO")]], [[("S", "R"), ("R", 1)]]), ([], [])]
    assert_command_with_fetch(circuit, "IN 1", assertions)

    assertions = [([[("E", "R"), ("R", 3)]], [[("S", "IO")]]), ([], []), ([], [])]
    assert_command_with_fetch(circuit, "OUT 3", assertions)
    assertions = [([[("E", "R"), ("R", 1)]], [[("S", "IO")]]), ([], []), ([], [])]
    assert_command_with_fetch(circuit, "OUT 1", assertions)


def test_io_unit(circuit):
    s, e = Lines("se")
    bus_ = bus()
    inputs = s >> e >> bus_
    ins, outs = (inputs >> circuit_module.IoUnit()).split()
    outs >>= bus_
    inputs >>= ins
    truth_table_test(
        inputs,
        outs,
        (
            ((0, 0, "3"), ("0", "3")),
            ((0, 0, "3", "42"), ("0", "3")),
            ((0, 0, "3"), ("0", "3")),
            ((1, 0, "3"), ("3", "3")),
            ((0, 0, "3"), ("3", "3")),
            ((1, 0, "3"), ("3", "3")),
            ((0, 0, "3"), ("3", "3")),
            ((0, 0, "52"), ("3", "52")),
            ((0, 0, "52"), ("3", "52")),
            ((1, 0, "52"), ("52", "52")),
            ((0, 0, "52"), ("52", "52")),
            ((0, 1, "52"), ("52", "52")),
            ((0, 0, "52"), ("52", "52")),
            ((1, 0, "42"), ("42", "42")),
        ),
        draw=1,
    )


def test_jmp_if_controller(circuit):
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
        ([[("E", "RAM")]], [[("S", "IAR")]]),
    ]
    assert_command_with_fetch(circuit, "JC 24", assertions, flags="C")
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
        ([[("E", "RAM")]], [[("S", "IAR")]]),
    ]
    assert_command_with_fetch(circuit, "JCA 24", assertions, flags="C")
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "JE 24", assertions, flags="C")
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "JE 24", assertions, flags="CAZ")
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
        ([[("E", "RAM")]], [[("S", "IAR")]]),
    ]
    assert_command_with_fetch(circuit, "JEZ 24", assertions, flags="CAZ")


def test_data_controller(circuit):
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "RAM")]], [[("S", "R"), ("R", 2)]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
    ]
    assert_command_with_fetch(circuit, "DATA 2", assertions)
    assertions = [
        ([[("E", "IAR")], [("E", "B1")]], [[("S", "MAR")], [("S", "ACC")]]),
        ([[("E", "RAM")]], [[("S", "R"), ("R", 3)]]),
        ([[("E", "ACC")]], [[("S", "IAR")]]),
    ]
    assert_command_with_fetch(circuit, "DATA 3", assertions)


def alu_tags(*enabled_tags):
    return [[("ALU", "OP"), ("ALU", tag)] for tag in enabled_tags]

def alu_s5():
    return [[("S", "FLAGS")]]

def test_add_controller(circuit):
    alus = []
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "ADD 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "ADD 2 2", assertions)


def test_shr_controller(circuit):
    alus = [2]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "SHR 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "SHR 2 2", assertions)


def test_shl_controller(circuit):
    alus = [1]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "SHL 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "SHL 2 2", assertions)


def test_not_controller(circuit):
    alus = [1, 2]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "NOT 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "NOT 2 2", assertions)


def test_and_controller(circuit):
    alus = [0]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "AND 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "AND 2 2", assertions)


def test_or_controller(circuit):
    alus = [0, 2]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "OR 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "OR 2 2", assertions)


def test_xor_controller(circuit):
    alus = [0, 1]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 1)]]),
    ]
    assert_command_with_fetch(circuit, "XOR 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([[("E", "ACC")]], [[("S", "R"), ("R", 2)]]),
    ]
    assert_command_with_fetch(circuit, "XOR 2 2", assertions)


def test_cmp_controller(circuit):
    alus = [0, 1, 2]
    assertions = [
        ([[("E", "R"), ("R", 1)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 3)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "CMP 3 1", assertions)
    assertions = [
        ([[("E", "R"), ("R", 2)]], [[("S", "TMP")]]),
        ([[("E", "R"), ("R", 2)]] + alu_tags(*alus), [[("S", "ACC")]] + alu_s5()),
        ([], []),
    ]
    assert_command_with_fetch(circuit, "CMP 2 2", assertions)


def test_alu_runner(circuit):
    with scope("StepperIn"):
        stepper = bus(circuit_module.Stepper.N_OUTS - 1)
    with scope("IrIn"):
        ir = bus(8)
    output = stepper >> ir >> circuit_module.AluRunner()
    inputs = stepper[3:6] >> ir[:4]
    truth_table_test(
        inputs,
        output,
        (
            ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
            ((1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
            ((0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
            ((0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
            ((0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
            ((1, 0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0)),
            ((1, 0, 0, 1, 1, 0, 0), (0, 0, 0, 1, 0, 0)),
            ((1, 0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 0, 0)),
            ((0, 1, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0)),
            ((0, 1, 0, 1, 1, 0, 0), (1, 0, 0, 0, 1, 0)),
            ((0, 1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 0)),
            ((0, 0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
            ((0, 0, 1, 1, 1, 0, 0), (0, 0, 0, 0, 0, 1)),
            ((0, 0, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),
        ),
        draw=1,
    )


def test_non_alu(circuit):
    with scope("StepperIn"):
        stepper = bus(circuit_module.Stepper.N_OUTS - 1)
    with scope("IrIn"):
        ir = bus(8)
    with scope("Flags"):
        flags = bus(4)
    output = stepper >> ir >> flags >> circuit_module.NonAluModule()
    inputs = stepper[3:5] >> ir[:4]
    truth_table_test(
        inputs,
        output[:4],
        (
            ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0)),
            ((1, 0, 1, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 1, 1, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0)),
            ((1, 0, 0, 0, 0, 0), (1, 0, 0, 0)),
            ((0, 1, 0, 0, 0, 0), (0, 1, 0, 0)),
            ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0)),
            ((1, 0, 0, 0, 0, 1), (0, 0, 1, 0)),
            ((0, 1, 0, 0, 0, 1), (0, 0, 0, 1)),
        ),
        draw=1,
    )
    truth_table_test(
        stepper[3:6] >> flags >> ir,
        output.typed(("INSTRUCTION", "J")),
        (
            ((1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0), (1, 0, 0)),
            ((0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0), (0, 1, 0)),
            ((0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0), (0, 0, 0)),
            ((0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0), (0, 0, 0)),
            ((0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0), (0, 0, 1)),
            ((0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0), (0, 0, 1)),
            ((0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0), (0, 0, 0)),
            ((0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0), (0, 0, 0)),
            ((0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0), (0, 0, 0)),
            ((0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1), (0, 0, 1)),
            ((0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0), (0, 0, 1)),
        ),
        draw=1,
    )


def test_types(circuit):
    a, b = Line("a") >> Line("b") >> circuit_module.with_type(("a",), "b")
    assert a.is_type("b")
    assert a.is_type(("a",))
    assert not a.is_type("a")
    assert not a.is_type(("b",))
    assert b.is_type("b")
    assert b.is_type(("a",))
    assert not b.is_type("a")
    assert not b.is_type(("b",))
    assert a.is_types("b", ("a",))
    assert a.is_types("b")
    assert not a.is_types("b", "c")
    assert not a.is_types("c")
    assert len(Lines(a, b).typed("a")) == 0
    assert set(Lines(a, b).typed("b")) == {a, b}
    assert set(Lines(a, b).typed("b", ("a",))) == {a, b}
    assert set(Lines(a, b).typed("b", ("c",))) == set()
    assert set(Lines(a, b, Line("a")).typed("b")) == {a, b}
    assert set(Lines(a, b).typed("b")) == {a, b}
    a.add_type("c")
    assert set(Lines(a, b).typed("b")) == {a, b}
    assert set(Lines(a, b).typed("c")) == {a}
    assert set(Lines(a, b).typed("c")) == {a}

    d, e = Line("d") >> Line("e")
    dt, et = d >> e >> circuit_module.with_type("d")
    assert d is dt
    assert e is et
