from typing import Iterable

from pytest import fixture
from pprint import pprint

from circuit import (
    Line,
    Lines,
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


def test_lines_lookup():
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


def test_string_lines():
    assert "".join(line.name for line in Lines("abc")) == "abc"


@fixture
def circuit():
    with new_circuit() as circuit:
        yield circuit


def truth_table_test(lines: Lines, test_output, truth_table):
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
        previous_state = simulate(feed_dict, previous_state=previous_state, draw=0)
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
    n_inputs = 5
    inputs = bus(n_inputs)
    outputs = inputs >> decoder
    assert len(outputs) == 2 ** n_inputs
    truth_table_test(
        inputs, outputs, (((str(i),), str(2 ** i)) for i in range(2 ** n_inputs))
    )


def test_ram(circuit):
    s = Line("s")
    e = Line("e")
    inputs = bus()
    sa = Line("sa")
    mar_inputs = bus()
    all_inputs = s >> e >> inputs >> sa >> mar_inputs
    all_inputs >> ram
    output = simulate({s: 1}, circuit)
    print(len(output))
    feed_dict = {i: 1 for i in inputs}
    feed_dict[s] = 0
    feed_dict[e] = 0
    feed_dict[sa] = 1
    for i in mar_inputs:
        feed_dict[i] = 1
    output = simulate(feed_dict, circuit, previous_state=output)
    feed_dict[s] = 0
    feed_dict[e] = 0
    feed_dict[sa] = 0
    output = simulate(feed_dict, circuit, previous_state=output)
    feed_dict[s] = 1
    output = simulate(feed_dict, circuit, previous_state=output)
    feed_dict[s] = 0
    # for i in inputs:
    #     del feed_dict[i]
    output = simulate(feed_dict, circuit, previous_state=output)
    feed_dict[e] = 0
    output = simulate(feed_dict, circuit, previous_state=output)

    

