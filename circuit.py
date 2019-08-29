from __future__ import annotations

import abc
import collections
import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from cached_property import cached_property
from typing_extensions import Deque
import enum


@contextmanager
def new_circuit():
    circuit = Circuit()
    _CircuitContext.circuits.append(circuit)
    yield circuit
    _CircuitContext.circuits.pop()


def circuit_or_default(circuit: Optional[Circuit]) -> Circuit:
    return circuit or default_circuit()


def default_circuit() -> Circuit:
    return _CircuitContext.circuit()


class _CircuitContext:
    circuits = collections.deque()

    @classmethod
    def circuit(cls):
        return cls.circuits[-1]


def _chain(
    self: Union[Line, Lines], other: Union[Line, Lines, Gate]
) -> Union[Lines, Gate]:
    if isinstance(other, Line) or isinstance(other, Lines):
        return Lines(self, other)
    if isinstance(other, Gate):
        return other(Lines(self))
    raise TypeError(
        f"Other value {other} of type {type(other)} not supported in chaining."
    )


@dataclass(frozen=True, init=False)
class Lines:
    lines: Tuple[Line]

    def __init__(self, *args: List[LinesSpec]):
        object.__setattr__(self, "lines", tuple(self._flatten(args)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Lines):
            raise TypeError("Cannot compare non-lines to lines")
        return self.lines == other.lines

    @cached_property
    def dict(self):
        return {line.name: line for line in self.lines}

    def __getitem__(self, name: str) -> Optional[Line]:
        return self.dict.get(name, None)

    def __str__(self):
        return " ".join(line.name for line in self)

    def __iter__(self) -> Iterable[Line]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __rshift__(self, other: Union[Line, Lines, Gate]) -> Union[Lines, Gate]:
        return _chain(self, other)

    def __ilshift__(self, other: Lines):
        self.tie(other)

    def tie(self, other: Line):
        if len(other) != len(self):
            raise TypeError(
                f"<<= operator requires equal length args: {len(self)} vs. {len(other)}"
            )
        for from_line, to_line in zip(other, self):
            to_line <<= from_line

    def _flatten(self, args: LinesSpec) -> List[Line]:
        if isinstance(args, str):
            return [Line(x) for x in args]
        if not args:
            return []
        if isinstance(args, Line):
            return [args]
        if isinstance(args, Iterable):
            return [x for arg in args for x in self._flatten(arg)]
        if isinstance(args, Lines):
            return list(self.lines)
        raise TypeError(f"Lines input {args} of type {type(args)} not supported.")


@dataclass(frozen=True, eq=False)
class Line:
    name: Optional[str] = None

    def __and__(self, other: Line) -> Line:
        return self >> other >> and_gate

    def __not__(self) -> Line:
        return self >> not_gate

    def __iter__(self) -> Iterable[Line]:
        yield self

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __or__(self, other: Line) -> Line:
        return self >> other >> or_gate

    def __rshift__(self, other: Union[Line, Lines, Gate]) -> Union[Lines, Gate]:
        return _chain(self, other)

    def tie(self, other: Line):
        if len(other) != 1:
            raise TypeError(
                f"<<= operator can only take one input: saw {len(other)} inputs for {other}."
            )
        other, = other
        _tie(other, self)

    def __ilshift__(self, other: Line):
        self.tie(other)


class Gate(abc.ABC):
    @abc.abstractmethod
    def run(self, in_lines: Lines) -> Lines:
        ...

    def __call__(self, lines: Lines) -> Lines:
        return Lines(self.run(lines))


@dataclass(frozen=True)
class Op:
    in_lines: Lines
    out_lines: Lines
    fn: Callable[CircuitState, CircuitState]

    @classmethod
    def create(
        cls,
        in_lines: LinesSpec,
        out_lines: LinesSpec,
        fn: Callable[CircuitState, LinesSpec],
    ) -> Op:
        return cls(Lines(in_lines), Lines(out_lines), fn)


def _if(lines: LinesSpec, true_value: bool = True) -> Lines:
    lines = Lines(lines)
    out_line = Line("_")

    def op(in_values: CircuitState) -> CircuitState:
        all_true = all(in_values[line] for line in lines)
        value = true_value if all_true else (not true_value)
        return {out_line: value}

    _push_op(Op.create(lines, out_line, op))
    return out_line


def _push_op(op: Op):
    default_circuit().push_op(op)


def _tie(from_line: Line, to_line: Line):
    def op(in_values: CircuitState) -> CircuitState:
        return {to_line: in_values[from_line]}

    _push_op(Op.create(from_line, to_line, op))


class NandGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return Lines(_if(in_lines, False))


class NotGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return [in_line >> in_line >> nand for in_line in in_lines]


class AndGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return in_lines >> nand >> not_gate


class OrGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return in_lines >> not_gate >> nand


class XorGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return (in_lines >> or_gate) >> (in_lines >> and_gate >> not_gate) >> and_gate


class BitGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        i, s = in_lines
        a = i >> s >> nand
        b = a >> s >> nand
        c = Line("c")
        o = c >> a >> nand
        c_out = o >> b >> nand
        c_out >> c >> tie
        return o


class ByteGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        in_lines = list(in_lines)
        input_lines, s = in_lines[:-1], in_lines[-1]
        return [input_line >> s >> bit for input_line in input_lines]


class Enabler(Gate):
    def run(self, in_lines: Lines) -> Lines:
        in_lines = list(in_lines)
        input_lines, e = in_lines[:-1], in_lines[-1]
        return [input_line >> e >> and_gate for input_line in input_lines]


class Register(Gate):
    def run(self, in_lines: Lines) -> Lines:
        in_lines = list(in_lines)
        input_lines, s, e = Lines(in_lines[:-2]), in_lines[-2], in_lines[-1]
        return input_lines >> s >> byte >> e >> enabler


class Decoder(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return [
            Lines(inputs) >> and_gate
            for inputs in itertools.product(*zip(in_lines, in_lines >> not_gate))
        ]


class Tie(Gate):
    def run(self, in_lines: Lines) -> Lines:
        in_lines = list(in_lines)
        from_lines = Lines(in_lines[: len(in_lines) // 2])
        to_lines = Lines(in_lines[len(in_lines) // 2 :])
        to_lines <<= from_lines


class RAM(Gate):
    def run(self, in_lines: Lines) -> Lines:
        in_lines = list(in_lines)
        s = in_lines[0]
        e = in_lines[1]
        bus = Lines(in_lines[2:10])
        sa = in_lines[10]
        mar_inputs = Lines(in_lines[11:19])
        mar_outputs = mar_inputs >> sa >> register
        mar_outputs = list(mar_outputs)
        row_decoder = Lines(mar_outputs[:4]) >> decoder
        col_decoder = Lines(mar_outputs[4:]) >> decoder
        for row in row_decoder:
            for col in col_decoder:
                x = row >> col >> and_gate
                bus >> (x >> s >> and_gate) >> (
                    x >> e >> and_gate
                ) >> register >> bus >> tie


nand = NandGate()
not_gate = NotGate()
and_gate = AndGate()
or_gate = OrGate()
xor_gate = XorGate()
bit = BitGate()
byte = ByteGate()
enabler = Enabler()
register = Register()
decoder = Decoder()
tie = Tie()
ram = RAM()


LinesSpec = Optional[Union[Line, str, Lines, Iterable[Any]]]
LinesSpec = Optional[Union[Line, str, Lines, Iterable[LinesSpec]]]


def inputs(lines_spec: LinesSpec) -> Lines:
    return Lines(lines_spec)


FeedDict = Dict[Line, bool]

Simulation = Dict[Line, bool]
CircuitState = Simulation


class Circuit:
    def __init__(self):
        self.inputs_to_ops: Dict[Line, Set[Op]] = collections.defaultdict(set)

    def push_op(self, op: Op):
        for input_line in op.in_lines:
            self.inputs_to_ops[input_line].add(op)

    def downstream_ops(self, line: Line) -> Optional[Op]:
        return self.inputs_to_ops[line]

    @property
    def lines(self) -> Lines:
        return Lines(
            line
            for op in self.ops
            for line_group in [op.in_lines, op.out_lines]
            for line in line_group
        )

    @property
    def ops(self) -> List[Op]:
        return list({op for ops in self.inputs_to_ops.values() for op in ops})


def constant_op(line: Line, value: bool) -> Op:
    return Op.create(None, line, lambda a: {line: value})


def bus(n_inputs=8) -> Lines:
    return Lines(map(str, range(n_inputs)))


def simulate(
    feed_dict: FeedDict,
    circuit: Optional[Circuit] = None,
    previous_state: Optional[CircuitState] = None,
    draw=False,
) -> Simulation:
    circuit = circuit_or_default(circuit)
    if not previous_state:
        previous_state = collections.defaultdict(bool)
    else:
        previous_state = dict(previous_state)
    while True:
        queue: Deque[Op] = collections.deque(
            constant_op(line, value) for line, value in feed_dict.items()
        )
        visited = set()
        stable = True
        while queue:
            op = queue.pop()
            if op in visited:
                continue
            visited.add(op)
            output_states = op.fn(previous_state)
            for line, value in output_states.items():
                stable &= value == previous_state[line]
                previous_state[line] = value
                queue.extend(circuit.downstream_ops(line))
        if stable:
            if draw:
                import graph_tools

                graph_tools.draw(circuit, dict(feed_dict), dict(previous_state))
            return dict(previous_state)
