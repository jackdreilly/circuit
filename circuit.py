import abc
import collections
import enum
import itertools
import json
import parser
import queue as queue_module
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pprint import pprint
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)

from cached_property import cached_property
from dataclasses_json import dataclass_json
from typing_extensions import Deque

BYTE_SIZE = 8

class Lines:
    ...
class Circuit:
    ...
class Line:
    ...
class Gate:
    ...
class LinesSpec:
    ...
class CircuitState:
    ...



class Clock:
    def __init__(self):
        with scope("Clock"):
            self.clk = Line("Clk")
            self.delay_clk = Line("DelayClk")
            self.timestep = 0
            self.clk_s, = self.clk >> self.delay_clk >> and_gate << "ClkS"
            self.clk_e, = self.clk >> self.delay_clk >> or_gate << "ClkE"

    def step(self):
        self.timestep += 1
        return self.current_step

    @property
    def lines(self) -> Lines:
        return self.clk >> self.clk_s >> self.clk_e

    @property
    def current_step(self):
        return {self.clk: self.clk_value, self.delay_clk: self.delay_clk_value}

    @property
    def clk_value(self):
        return self.timestep % 4 in (1, 2)

    @property
    def delay_clk_value(self):
        return (self.timestep % 4) > 1


class ScopeStack:
    stack = collections.deque()
    memory = collections.defaultdict(int)

    @classmethod
    def reset(cls):
        cls.stack = collections.deque()
        cls.memory = collections.defaultdict(int)

    @classmethod
    def push(cls, value: str, reenter=False):
        cls.stack.append(value)
        proposal = cls.string()
        priors = cls.memory[proposal]
        cls.memory[proposal] += 1
        if priors and not reenter:
            cls.pop()
            cls.push(f"{value}-{priors}")

    @classmethod
    def pop(cls) -> str:
        return cls.stack.pop()

    @classmethod
    def string(cls) -> str:
        return "/".join(cls.stack)


@contextmanager
def scope(value: str) -> ScopeStack:
    ScopeStack.push(value)
    yield ScopeStack
    ScopeStack.pop()


def scoped(value: str) -> Lines:
    ScopeStack.push(value)
    return Lines()


def unscoped() -> Lines:
    ScopeStack.pop()
    return Lines()


@contextmanager
def scope_pop() -> ScopeStack:
    value = ScopeStack.pop()
    yield ScopeStack
    ScopeStack.push(value, reenter=True)


def active_name() -> str:
    return ScopeStack.string()


def rescope(lines: Lines):
    for line in lines:
        line << line.name[line.name.rfind("/") + 1 :]


@contextmanager
def new_circuit():
    ScopeStack.reset()
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
    self: Union[Line, Lines], other: Union[Line, Lines, Gate, LinesSpec]
) -> Union[Lines, Gate]:
    if isinstance(other, Line) or isinstance(other, Lines):
        return Lines(self, other)
    if isinstance(other, Gate):
        return other(Lines(self))
    if isinstance(other, Iterable):
        return Lines(self, other)
    raise TypeError(
        f"Other value {other} of type {type(other)} not supported in chaining."
    )


def _bit(s: str) -> int:
    return 2 ** len(s) - int(s, 2) - 1


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


@dataclass(frozen=True, init=False)
class Lines:
    lines: Tuple[Line]

    def map(self, fn, chunk_size=1) -> Lines:
        with scope("Map"):
            return Lines(
                [
                    scoped(str(i)) >> fn(i, l) >> with_type(("MAP", i)) >> unscoped()
                    for i, l in enumerate(chunker(self, chunk_size))
                ]
            )

    def zipped(self, fn) -> Lines:
        with scope("Zip"):
            return Lines(
                [
                    scoped(str(i)) >> fn(i, a, b) >> with_type(("ZIP", i)) >> unscoped()
                    for i, (a, b) in enumerate(self.zip)
                ]
            )

    def bit(self, b: str) -> Line:
        return self[_bit(b)]

    def typed(self, *line_types) -> Lines:
        return Lines(line for line in self if line.is_types(*line_types))

    @property
    def zip(self):
        return zip(*self.split())

    def pop(self, amount=1) -> Tuple[Union[Lines, Line], Lines]:
        l = list(self)
        ret_value = Lines(l[:amount]).or_line
        rest = Lines(l[amount:]).or_line
        return ret_value, rest

    @property
    def or_line(self) -> Union[Line, Lines]:
        return self.line or self

    def split(self, *split_args) -> Tuple[Lines]:
        if not split_args:
            split_args = (len(self) // 2,)
        returns = []
        lines = self
        for split in split_args:
            v, lines = lines.pop(split)
            returns.append(v)
        returns.append(lines)
        return tuple(returns)

    def __init__(self, *args: List[LinesSpec]):
        object.__setattr__(self, "lines", tuple(self._flatten(args)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Lines):
            raise TypeError("Cannot compare non-lines to lines")
        return self.lines == other.lines

    @cached_property
    def dict(self):
        return {line.name: line for line in self.lines}

    def __getitem__(self, key: str) -> Optional[Union[Line, Lines]]:
        if isinstance(key, slice):
            return Lines(self.lines[key])
        if isinstance(key, int):
            return self.lines[key]
        return self.dict.get(key, None)

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

    @property
    def line(self) -> Optional[Line]:
        if len(self) == 1:
            return self.lines[0]

    def __lshift__(self, name: str):
        for line in self:
            line << name
        return self

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


LineValue = Union[bool, int]



@dataclass_json
@dataclass(frozen=False, eq=False)
class Line:
    name: Optional[str] = None
    default_value: LineValue = 0
    is_blocking: bool = True
    line_types: Set[Any] = field(default_factory=set)

    @property
    def as_json(self):
        v = json.loads(self.to_json())
        v["line_types"] = list(map(str, v["line_types"]))
        return v

    def is_type(self, line_type: Any) -> bool:
        return line_type in self.line_types

    def is_types(self, *line_types) -> bool:
        return all(self.is_type(line_type) for line_type in line_types)

    def add_type(self, line_type: Any):
        self.line_types.add(line_type)

    def add_types(self, line_types: Iterable[Any]):
        self.line_types.update(line_types)

    def __post_init__(self):
        self << (self.name or "line")

    def set_name(self, name: str):
        object.__setattr__(self, "name", name)

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

    def __irshift__(self, other: Union[Line, Lines, Gate]) -> Union[Lines, Gate]:
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

    def __lshift__(self, name: str):
        with scope(name):
            self.set_name(active_name())
        return self


class Gate(abc.ABC):
    output_scope = "outputs"

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Gate", "")

    def rescope_outputs(self, outputs: Lines):
        pass

    @abc.abstractmethod
    def run(self, in_lines: Lines) -> Lines:
        ...

    def __call__(self, lines: Lines) -> Lines:
        with scope(self.name):
            outputs = Lines(self.run(lines))
            if self.output_scope:
                with scope("outputs"):
                    rescope(outputs)
                    self.rescope_outputs(outputs)
        return outputs


@dataclass(frozen=True)
class Op:
    in_lines: Lines
    out_lines: Lines
    fn: Callable[[CircuitState], CircuitState]
    name: str

    @classmethod
    def create(
        cls,
        in_lines: LinesSpec,
        out_lines: LinesSpec,
        fn: Callable[[CircuitState], LinesSpec],
        name: str = "op",
    ):
        with scope(name):
            return cls(Lines(in_lines), Lines(out_lines), fn, active_name())

    @property
    def as_json(self):
        return {"op_type": "if"}


def _if(lines: LinesSpec, true_value: bool = True) -> Lines:
    lines = Lines(lines)
    with scope_pop():
        out_line = Line("x")

    def op(in_values: CircuitState) -> CircuitState:
        all_true = all(in_values[line] for line in lines)
        value = true_value if all_true else (not true_value)
        return {out_line: value}

    _push_op(Op.create(lines, out_line, op, "if"))
    return out_line


def _push_op(op: Op):
    default_circuit().push_op(op)


def _tie(from_line: Line, to_line: Line):
    default_circuit().tie(from_line, to_line)


class SplitGate(Gate):
    split_op: Gate

    def run(self, in_lines: Lines) -> Lines:
        return in_lines.zipped(lambda i, a, b: a >> b >> self.split_op)


class MapGate(Gate):
    map_op: Gate

    def run(self, in_lines: Lines) -> Lines:
        return in_lines.map(lambda i, l: l >> self.map_op)


class PassGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return _if(in_lines, True)


class NandGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return Lines(_if(in_lines, False))


class NotGate(Gate):
    def run(self, in_line: Lines) -> Lines:
        return in_line >> in_line >> nand


class Notter(MapGate):
    map_op = NotGate()


class AndGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return in_lines >> nand >> not_gate


class OrGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return in_lines >> not_gate >> nand


class XorGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return (in_lines >> or_gate) >> (in_lines >> nand) >> and_gate


class BitGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        i, s = in_lines
        a = i >> s >> nand
        b = a >> s >> nand
        c = Line("c", default_value=1, is_blocking=False)
        o = (c >> a >> nand) << "o"
        c_out = o >> b >> nand
        c_out >> c >> tie
        return o >> with_type("BIT")


class ByteGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        s, input_lines = in_lines.pop()
        return input_lines.map(lambda i, l: l >> s >> bit)


class Enabler(Gate):
    def run(self, in_lines: Lines) -> Lines:
        e, input_lines = in_lines.pop()
        return input_lines.map(lambda i, l: l >> e >> and_gate)


class Register(Gate):
    def run(self, in_lines: Lines) -> Lines:
        s, e, input_lines = in_lines.split(1, 1)
        return e >> (s >> input_lines >> byte) >> enabler >> with_type("ENABLER")


class Decoder(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return Lines(itertools.product(*zip(in_lines, in_lines >> not_gate))).map(
            lambda i, a: a >> and_gate, chunk_size=len(in_lines)
        )


class Tie(Gate):
    output_scope = None

    def run(self, in_lines: Lines) -> Lines:
        from_lines, to_lines = in_lines.split()
        to_lines <<= from_lines
        return from_lines


class RAM(Gate):
    MEM_SIZE = BYTE_SIZE

    def run(self, in_lines: Lines) -> Lines:
        s, e, sa, rest = in_lines.split(1, 1, 1)
        bus, mar_inputs = rest.split()
        with scope("MAR"):
            mar_outputs = (
                sa >> mar_inputs[-self.MEM_SIZE :] >> byte >> with_type("MAROUTPUT")
            )
        with scope("Decoder"):
            rows, cols = mar_outputs.pop(len(mar_outputs) // 2)
            with scope("RowDecoder"):
                row_decoder = (rows >> decoder)[::-1]
            with scope("ColDecoder"):
                col_decoder = (cols >> decoder)[::-1]
        with scope("Registers"):

            def mapper(i, rc):
                with scope(f"RAMRegister-{i}"):
                    rcs = rc >> and_gate
                    return (
                        (rcs >> s >> and_gate)
                        >> (rcs >> e >> and_gate)
                        >> bus
                        >> register
                        >> bus
                        >> tie
                    )

            return Lines(itertools.product(row_decoder, col_decoder)).map(
                mapper, chunk_size=2
            )


class RShift(Gate):
    def run(self, in_lines: Lines) -> Lines:
        shift_in, lines, shift_out = in_lines.split(1, -1)
        return (shift_out >> shift_in >> lines) >> passer


class LShift(Gate):
    def run(self, in_lines: Lines) -> Lines:
        shift_in, shift_out, lines = in_lines.split(1, 1)
        return (shift_out >> lines >> shift_in) >> passer


nand = NandGate()
notter = Notter()
not_gate = notter
and_gate = AndGate()
or_gate = OrGate()
xor_gate = XorGate()
pass_gate = PassGate()


class Passer(MapGate):
    map_op = pass_gate


class Ander(SplitGate):
    split_op = and_gate


class Orer(SplitGate):
    split_op = or_gate


class Xorer(SplitGate):
    split_op = xor_gate


class SumGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        a, b, carry_in = in_lines
        c = a >> b >> xor_gate
        d = a >> b >> and_gate
        e = c >> carry_in >> and_gate
        carry_out = d >> e >> or_gate
        sum_gate = carry_in >> c >> xor_gate
        carry_out << "carry-out"
        sum_gate << "sum"
        return sum_gate >> carry_out


class Adder(Gate):
    def run(self, in_lines: Lines) -> Lines:
        carry_in, lines = in_lines.pop()
        sums = Lines()
        carry_out = carry_in
        for aa, bb in list(lines.zip)[::-1]:
            sum_line, carry_out = aa >> bb >> carry_out >> sum_gate
            sums >>= sum_line
        return carry_out >> Lines(sums)[::-1]

    def rescope_outputs(self, outputs: Lines) -> Lines:
        with scope("Sums"):
            rescope(outputs[1:])


class ComparatorGate(Gate):
    output_scope = None

    def run(self, in_lines: Lines) -> Lines:
        a, b, all_equal, a_larger = in_lines
        unequal = a >> b >> xor_gate
        equal = unequal >> not_gate
        all_equal_so_far = all_equal >> equal >> and_gate
        a_larger_out = all_equal >> unequal >> a >> and_gate >> a_larger >> or_gate
        a_larger_out << "ALarger"
        all_equal_so_far << "EqualSoFar"
        unequal << "Unequal"
        return all_equal_so_far >> a_larger_out >> unequal


class Comparator(Gate):
    output_scope = None

    def run(self, in_lines: Lines) -> Lines:
        equal_so_far = Line("EqualSoFarInitial", default_value=1)
        a_larger = Line("ALargerInitial", default_value=0)
        cs = Lines()
        for aa, bb in in_lines.zip:
            equal_so_far, a_larger, c = (
                aa >> bb >> equal_so_far >> a_larger >> comp_gate
            )
            cs >>= c
        return (
            (equal_so_far << "Is-Equal")
            >> (a_larger << "A-Is-Larger")
            >> (scoped("OutComparisons") >> cs << "c" >> unscoped())
        )


class RegisterSelector(Gate):
    output_scope = None

    def run(self, in_lines: Lines) -> Lines:
        clk_e, clk_s, ea, eb, sb, _, _, ir = in_lines.split(1, 1, 1, 1, 1, 1, 3)
        a_lines, b_lines = ir.split()
        a_decoder, b_decoder = (a_lines >> decoder)[::-1], (b_lines >> decoder)[::-1]
        with scope("Enablers"):
            with scope("A"):
                ra = a_decoder.map(lambda i, l: clk_e >> l >> ea >> and_gate)
            with scope("B"):
                rb = b_decoder.map(lambda i, l: clk_e >> l >> eb >> and_gate)
            with scope("Combiner"):
                re = (ra >> rb).zipped(
                    lambda i, a, b: a >> b >> or_gate >> with_type(("R", i))
                ) >> with_type(("E", "R"), "ENABLER")
        with scope("Selectors"):
            rs = b_decoder.map(
                lambda i, l: clk_s >> l >> sb >> and_gate >> with_type(("R", i))
            ) >> with_type(("S", "R"), "SELECTOR")
        return re >> rs >> with_type("CONTROL")


class AluRunner(Gate):
    def run(self, in_lines: Lines) -> Lines:
        stepper, ir = in_lines.pop(Stepper.N_OUTS - 1)
        with scope("ALU"):
            alus = (
                ir[1:4].map(
                    lambda i, l: stepper[4]
                    >> ir[0]
                    >> l
                    >> and_gate
                    >> with_type(("ALU", i))
                )
                >> with_type("ENABLER", "CONTROL", ("ALU", "OP"))
                << "Alu"
            )
        outs = Lines()
        outs >>= ir[0] >> stepper[3] >> and_gate >> with_type(("E", "RB"), ("S", "TMP"))
        outs >>= (
            ir[0]
            >> stepper[4]
            >> and_gate
            >> with_type(("E", "RA"), ("S", "ACC"), ("S", "FLAGS"))
        )
        outs >>= (
            ir[0]
            >> stepper[5]
            >> (ir[1:4] >> and_gate >> not_gate)
            >> and_gate
            >> with_type(("E", "ACC"), ("S", "RB"))
        )
        outs << "Phase"
        return alus >> outs


class NonAluModule(Gate):
    def run(self, in_lines: Lines) -> Lines:
        stepper, ir, flags = in_lines.split(Stepper.N_OUTS - 1, 8)
        not_0 = ir[0] >> not_gate
        with scope("Decoder"):
            decoded = (ir[1:4] >> decoder).map(lambda i, l: not_0 >> l >> and_gate)
        with scope("Load"):
            outs = (
                decoded.bit("000")
                >> stepper[3]
                >> and_gate
                >> with_type(("E", "RA"), ("S", "MAR"))
            )
            outs >>= (
                decoded.bit("000")
                >> stepper[4]
                >> and_gate
                >> with_type(("E", "RAM"), ("S", "RB"))
            )
        with scope("Store"):
            outs >>= (
                decoded.bit("001")
                >> stepper[3]
                >> and_gate
                >> with_type(("E", "RA"), ("S", "MAR"))
            )
            outs >>= (
                decoded.bit("001")
                >> stepper[4]
                >> and_gate
                >> with_type(("E", "RB"), ("S", "RAM"))
            )
        with scope("Data"):
            outs >>= (
                decoded.bit("010")
                >> stepper[3]
                >> and_gate
                >> with_type(("E", "B1"), ("E", "IAR"), ("S", "MAR"), ("S", "ACC"))
            )
            outs >>= (
                decoded.bit("010")
                >> stepper[4]
                >> and_gate
                >> with_type(("E", "RAM"), ("S", "RB"))
            )
            outs >>= (
                decoded.bit("010")
                >> stepper[5]
                >> and_gate
                >> with_type(("E", "ACC"), ("S", "IAR"))
            )
        with scope("Jump"):
            with scope("Register"):
                outs >>= (
                    decoded.bit("011")
                    >> stepper[3]
                    >> and_gate
                    >> with_type(("E", "RB"), ("S", "IAR"))
                )
            with scope("Address"):
                outs >>= (
                    decoded.bit("100")
                    >> stepper[3]
                    >> and_gate
                    >> with_type(("E", "IAR"), ("S", "MAR"))
                )
                outs >>= (
                    decoded.bit("100")
                    >> stepper[4]
                    >> and_gate
                    >> with_type(("E", "RAM"), ("S", "IAR"))
                )
            with scope("If"):
                outs >>= (
                    decoded.bit("101")
                    >> stepper[3]
                    >> and_gate
                    >> with_type(("E", "B1"), ("E", "IAR"), ("S", "MAR"), ("S", "ACC"))
                    >> with_type(("INSTRUCTION", "J"))
                )
                outs >>= (
                    decoded.bit("101")
                    >> stepper[4]
                    >> and_gate
                    >> with_type(("E", "ACC"), ("S", "IAR"))
                    >> with_type(("INSTRUCTION", "J"))
                )
                outs >>= (
                    decoded.bit("101")
                    >> stepper[5]
                    >> (
                        (flags >> ir[4:]).zipped(
                            lambda i, f, ir_l: f >> ir_l >> and_gate
                        )
                        >> or_gate
                    )
                    >> and_gate
                    >> with_type(("E", "RAM"), ("S", "IAR"))
                    >> with_type(("INSTRUCTION", "J"))
                )
        with scope("Clear"):
            outs >>= (
                decoded.bit("110")
                >> stepper[3]
                >> and_gate
                >> with_type(("E", "B1"), ("S", "FLAGS"))
            )
        with scope("IO"):
            outs >>= (
                decoded.bit("111")
                >> stepper[3]
                >> ir[4]
                >> and_gate
                >> with_type(("E", "RB"), ("S", "IO"))
            )
            outs >>= (
                decoded.bit("111")
                >> stepper[4]
                >> (ir[4] >> not_gate)
                >> and_gate
                >> with_type(("E", "IO"), ("S", "RB"))
            )
        return outs


class ZeroGate(Gate):
    def run(self, in_lines: Lines) -> Lines:
        return in_lines >> or_gate >> not_gate


class Bus1(Gate):
    def run(self, in_lines: Lines) -> Lines:
        s, lines = in_lines.pop()
        and_lines, or_line = lines.split(-1)
        not_s = s >> not_gate
        return and_lines.map(lambda i, l: l >> not_s >> and_gate) >> (
            or_line >> s >> or_gate
        )


class Alu(Gate):
    def run(self, in_lines: Lines) -> Lines:
        op, carry_in, lines = in_lines.split(3, 1)
        a, b = lines.split()
        with scope("OpDecoder"):
            op_decoder = op >> decoder
        equal, a_larger, op_cmp = (a >> b >> comp).split(1, 1)
        op_xorer = lines >> xorer
        op_orer = lines >> orer
        op_ander = lines >> ander
        op_notter = lines >> notter
        lshifter_out, lshifter = (carry_in >> a >> lshift).pop()
        rshifter_out, rshifter = (carry_in >> a >> rshift).pop()
        adder_out, op_adder = (carry_in >> a >> b >> adder).pop()
        ops = [
            op_cmp,
            op_xorer,
            op_orer,
            op_ander,
            op_notter,
            lshifter,
            rshifter,
            op_adder,
        ]
        with scope("Enabler"):
            enabled_ops = [
                decode_line >> gate >> enabler
                for gate, decode_line in zip(ops, op_decoder)
            ]
        cs = Lines()
        with scope("Bundler"):
            for i, bundles in enumerate(zip(*enabled_ops)):
                with scope(str(i)):
                    cs >>= Lines(bundles) >> or_gate
        out_zero = cs >> zero_gate << "Zero"
        with scope("CarryOuts"):
            lshifter_out = lshifter_out >> op_decoder[-3] >> and_gate
            rshifter_out = rshifter_out >> op_decoder[-2] >> and_gate
            adder_out = adder_out >> op_decoder[-1] >> and_gate
            carry_out = lshifter_out >> rshifter_out >> adder_out >> or_gate
        cs << "Comparator"
        carry_out << "CarryOut"
        return a_larger >> equal >> out_zero >> carry_out >> cs

    def rescope_outputs(self, outputs: Lines):
        with scope("Comparator"):
            rescope(outputs[4:])


class ControlSection(Gate):
    output_scope = None

    def run(self, in_lines: Lines) -> Lines:
        clk, clk_s, clk_e, ir, flags, stepper_ = in_lines.split(1, 1, 1, 8, 4)
        outs = Lines()
        phases = Lines()
        with scope("Fetch"):
            phases >>= stepper_[0] >> with_type(
                ("E", "B1"), ("E", "IAR"), ("S", "MAR"), ("S", "ACC")
            )
            phases >>= stepper_[1] >> with_type(("E", "RAM"), ("S", "IR"))
            phases >>= stepper_[2] >> with_type(("E", "ACC"), ("S", "IAR"))
        with scope("AluInstructions"):
            alus, phases_ = (stepper_ >> ir >> AluRunner()).pop(3)
            phases >>= phases_
            outs >>= alus
        with scope("NonAluInstructions"):
            phases >>= stepper_ >> ir >> flags >> NonAluModule()
        with scope("Selectors"):
            for type_tag in ("MAR", "IAR", "ACC", "RAM", "TMP", "FLAGS", "IR", "IO"):
                with scope(type_tag):
                    outs >>= (
                        phases.typed(("S", type_tag))
                        >> or_gate
                        >> clk_s
                        >> and_gate
                        >> with_type(("S", type_tag), "CONTROL", "SELECTOR")
                    )
        with scope("Enablers"):
            for type_tag in ("IAR", "RAM", "ACC", "IO"):
                with scope(type_tag):
                    outs >>= (
                        phases.typed(("E", type_tag))
                        >> or_gate
                        >> clk_e
                        >> and_gate
                        >> with_type(("E", type_tag), "CONTROL", "ENABLER")
                    )
            type_tag = "B1"
            with scope(type_tag):
                outs >>= (
                    phases.typed(("E", type_tag))
                    >> or_gate
                    >> with_type(("E", type_tag), "CONTROL", "ENABLER")
                )
        with scope("RegisterSelectors"):
            outs >>= (
                clk_e
                >> clk_s
                >> (phases.typed(("E", "RA")) >> or_gate)
                >> (phases.typed(("E", "RB")) >> or_gate)
                >> (phases.typed(("S", "RB")) >> or_gate)
                >> ir
                >> RegisterSelector()
            )
        return outs


class IoUnit(Gate):
    def run(self, in_lines: Lines) -> Lines:
        s, e, bus_ = in_lines.split(1, 1)
        with scope("In"):
            io_in = (bus() >> with_type("IO", ("IO", "IN"))).map(
                lambda i, l: l >> with_type(("IO", i))
            )
            e >> io_in >> enabler >> bus_ >> tie
        with scope("Out"):
            io_out = (s >> bus_ >> byte >> with_type("IO", ("IO", "OUT"))).map(
                lambda i, l: l >> with_type(("IO", i))
            )
        return io_in >> io_out


class Cpu(Gate):
    N_REGISTERS = 4

    def run(self, in_lines: Lines) -> Lines:
        clocks = in_lines
        bus_ = bus() >> with_type("MAINBUS")
        stepper_ = clocks[0] >> stepper
        with scope("IR"):
            ir_s = Line("IrS", is_blocking=False)
            ir = (ir_s >> bus_ >> byte).map(
                lambda i, l: l >> with_type(("IR", i), "IR")
            )
        with scope("ControlFlags"):
            flags = Lines(Line(s, is_blocking=False) for s in "CAEZ") >> with_type(
                "FLAG"
            )
            carry_in = flags[0]
        controllers = clocks >> ir >> flags >> stepper_ >> ControlSection()
        controllers.typed(("S", "IR")) >> ir_s >> tie
        with scope("IAR"):
            controllers.typed(("S", "IAR")) >> controllers.typed(
                ("E", "IAR")
            ) >> bus_ >> register >> with_type("IAR") >> bus_ >> tie
        with scope("Registers"):
            for i in range(self.N_REGISTERS):
                with scope(str(i)):
                    s = controllers.typed(("S", "R"), ("R", i))
                    e = controllers.typed(("E", "R"), ("R", i))
                    reg = s >> e >> bus_ >> register >> with_type(("CPUREG", i))
                    reg >> bus_ >> tie
        with scope("RAM"):
            controllers.typed(("S", "RAM")) >> controllers.typed(
                ("E", "RAM")
            ) >> controllers.typed(("S", "MAR")) >> bus_ >> bus_ >> ram
        with scope("BInput"):
            with scope("Tmp"):
                tmp = controllers.typed(("S", "TMP")) >> bus_ >> byte
            b = controllers.typed(("E", "B1")) >> tmp >> bus1
        with scope("ALU"):
            a_larger, equal, out_zero, carry_out, cs = (
                controllers.typed(("ALU", "OP")) >> carry_in >> bus_ >> b >> alu
            ).split(1, 1, 1, 1)
            caez = (
                controllers.typed(("S", "FLAGS"))
                >> carry_out
                >> a_larger
                >> equal
                >> out_zero
                >> byte
            )
            caez >> flags >> tie
        with scope("ACC"):
            controllers.typed(("S", "ACC")) >> controllers.typed(
                ("E", "ACC")
            ) >> cs >> register >> bus_ >> tie
        io_lines = (
            controllers.typed(("S", "IO"))
            >> controllers.typed(("E", "IO"))
            >> bus_
            >> IoUnit()
        )
        return io_lines


class Stepper(Gate):
    N_OUTS = 7

    def run(self, in_lines: Lines) -> Lines:
        clk = in_lines
        reset = Line("Reset", default_value=1, is_blocking=False)
        with scope("NotClock"):
            n_clk = clk >> not_gate
        with scope("(Reset|Clock)"):
            r_clk = clk >> reset >> or_gate
        with scope("(Reset|NotClock)"):
            r_n_clk = n_clk >> reset >> or_gate
        with scope("NotReset"):
            n_reset = reset >> not_gate
        with scope("Ms"):
            ms = n_reset
            for i in range(self.N_OUTS):
                for j, s in enumerate((r_n_clk, r_clk)):
                    with scope(str(i * 2 + j)):
                        ms >>= ms[-1] >> s >> bit
        with scope("Steps"):
            with scope("0"):
                steps = reset >> (ms[2] >> not_gate) >> or_gate
            for i in range(self.N_OUTS - 1):
                with scope(str(i + 1)):
                    steps >>= (
                        ms[(i + 1) * 2] >> (ms[(i + 2) * 2] >> not_gate) >> and_gate
                    )
        steps[-1] >> reset >> tie
        return steps[:-1] >> with_type("STEPPER")


controller = ControlSection()
stepper = Stepper()
zero_gate = ZeroGate()
comp_gate = ComparatorGate()
comp = Comparator()
bit = BitGate()
byte = ByteGate()
enabler = Enabler()
register = Register()
decoder = Decoder()
tie = Tie()
ram = RAM()
lshift = LShift()
rshift = RShift()
ander = Ander()
orer = Orer()
notter = Notter()
xorer = Xorer()
sum_gate = SumGate()
adder = Adder()
alu = Alu()
passer = Passer()
bus1 = Bus1()
cpu = Cpu()

LinesSpec = Optional[Union[Line, str, Lines, Iterable[Any]]]
LinesSpec = Optional[Union[Line, str, Lines, Iterable[LinesSpec]]]


def inputs(lines_spec: LinesSpec) -> Lines:
    return Lines(lines_spec)


FeedDict = Dict[Line, bool]

Simulation = Dict[Line, bool]
CircuitState = Simulation


class TieOp:
    def __init__(self, to_line: Line):
        self.to_line = to_line
        self._from_lines = set()
        self.name = "/".join([to_line.name, "tie"])

    @property
    def as_json(self):
        return {"op_type": "tie"}

    @cached_property
    def in_lines(self) -> Lines:
        return Lines(self.from_lines)

    @cached_property
    def out_lines(self) -> Lines:
        return Lines(self.to_line)

    def add_from_line(self, from_line: Line):
        self.clear_cache()
        self._from_lines.add(from_line)

    def clear_cache(self):
        for tag in ("in_lines", "out_lines", "from_lines"):
            self.__dict__.pop(tag, None)

    def fn(self, in_values: CircuitState) -> CircuitState:
        return {
            self.to_line: any(in_values[from_line] for from_line in self.from_lines)
        }

    @cached_property
    def from_lines(self):
        return set(self._from_lines)


class Circuit:
    def __init__(self):
        self.inputs_to_ops: Dict[Line, Set[Op]] = collections.defaultdict(set)
        self.outputs_to_ops: Dict[Line, Set[Op]] = collections.defaultdict(set)
        self.tie_ops: Dict[Line, Op] = keydefaultdict(TieOp)
        self._feed_ids = set()
        self._feed_lines = Lines()

    def push_op(self, op: Op):
        self.clear_cache()
        for input_line in op.in_lines:
            self.inputs_to_ops[input_line].add(op)
        for output_line in op.out_lines:
            self.outputs_to_ops[output_line].add(op)

    def tie(self, from_line: Line, to_line: Line):
        self.clear_cache()
        self.tie_ops[to_line].add_from_line(from_line)
        self.push_op(self.tie_ops[to_line])

    def clear_cache(self):
        for tag in ("sources", "lines", "ops"):
            self.__dict__.pop(tag, None)

    def downstream_ops(self, line: Line) -> Set[Op]:
        return self.inputs_to_ops[line]

    @cached_property
    def sources(self) -> Lines:
        return Lines(
            line
            for line in self.lines
            if (not line.is_blocking)
            or (line not in self.tie_ops and line not in self.outputs_to_ops)
        )

    @cached_property
    def lines(self) -> Lines:
        return Lines(
            line
            for op in self.ops
            for line_group in [op.in_lines, op.out_lines]
            for line in line_group
        )

    @cached_property
    def ops(self) -> List[Op]:
        return list({op for ops in self.inputs_to_ops.values() for op in ops})

    def set_feed(self, lines: Lines):
        ids = {id(line) for line in lines}
        if ids == self._feed_ids:
            return
        self.__dict__.pop("op_order", None)
        self._feed_lines = lines
        self._feed_ids = ids

    @cached_property
    def op_order(self):
        order = []
        queue = collections.deque()
        visited = set()
        visited_lines = set(self._feed_lines >> self.sources)
        for source in visited_lines:
            for op in self.downstream_ops(source):
                if all(
                    in_line in visited_lines or not in_line.is_blocking
                    for in_line in op.in_lines
                ):
                    queue.append(op)
        while queue:
            op = queue.popleft()
            if op in visited:
                continue
            order.append(op)
            visited.add(op)
            for line in op.out_lines:
                visited_lines.add(line)
                for op in self.downstream_ops(line):
                    if op in visited:
                        continue
                    if all(
                        in_line in visited_lines or not in_line.is_blocking
                        for in_line in op.in_lines
                    ):
                        queue.appendleft(op)
        return order


def bus(n_inputs=None) -> Lines:
    with scope("bus"):
        return Lines(
            Line(str(i), is_blocking=False) >> with_type("BUS", i)
            for i in range(n_inputs or BYTE_SIZE)
        )


class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class TypedOp(Gate):
    def __init__(self, line_types: Any):
        self.line_types = line_types

    def run(self, in_lines: Lines):
        pass

    def __call__(self, in_lines: Lines):
        for line in in_lines:
            line.add_types(self.line_types)
        return in_lines


def with_type(*line_types) -> Op:
    return TypedOp(line_types)


def bootloader_program_txt():
    return """
    IN   2     ;  # Load length of input program in Reg 2
    DATA 0  14 ;  # Load input program (14) starting register address into Reg 0
    ADD  0  2  ;  # Add length of input program to starting location to give Input loop stopping condition.
    DATA 1  1  ;  # Store "1" in register 1 (used for incrementing loop var).
    IN   3     ;  # START LOOP(Line 6) - Read next line of input program from input.
    ST   0  3  ;  # Store program line in next available RAM address
    ADD  1  0  ;  # Increment next available RAM address
    CMP  2  0  ;  # Compare next RAM address w/ the stopping condition
    JE   14    ;  # If Equal flag set, then we've read the whole program, jump to program.
    JMP  6     ;  # END_LOOP(Line 6) - Go and read more of the program.
"""


def bootloader_program():
    return parser.parse(bootloader_program_txt())


@dataclass(order=True)
class PrioritizedItem:
    priority: str
    item: Any=field(compare=False)

def simulate(
    feed_dict: FeedDict,
    circuit: Optional[Circuit] = None,
    previous_state: Optional[CircuitState] = None,
) -> Simulation:
    circuit = circuit_or_default(circuit)
    seed_state = feed_dict
    if not previous_state:
        seed_state = {source: source.default_value for source in circuit.sources}
        seed_state.update(feed_dict)
        previous_state = {}
    dirty_ops = set()
    queue = collections.deque()
    visited_lines = set()

    blocked_queue = queue_module.PriorityQueue()
    blocked = set()

    def update(op_output: CircuitState, block_feed_dict=True):
        for line, value in sorted(op_output.items(), key=lambda a: a[0].name):
            if block_feed_dict and (line in feed_dict):
                continue
            if line not in previous_state or previous_state[line] != value:
                for op in sorted(circuit.downstream_ops(line), key=lambda op: op.name):
                    if op in dirty_ops:
                        continue
                    queue.append(op)
                    dirty_ops.add(op)
            previous_state[line] = value
            visited_lines.add(line)
    
    update(seed_state, False)

    while queue or blocked:
        if queue:
            op = queue.popleft()
            if any(
                line not in visited_lines and line.is_blocking for line in op.in_lines
            ):
                if op in blocked:
                    continue
                blocked.add(op)
                blocked_queue.put(PrioritizedItem(op.name, op))
                continue
        else:
            op = blocked_queue.get().item
            blocked.remove(op)
        update(
            op.fn(
                {
                    line: previous_state.get(line, line.default_value)
                    for line in op.in_lines
                }
            )
        )
        dirty_ops.remove(op)
    return previous_state
