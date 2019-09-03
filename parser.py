from typing import List, Optional, Tuple


def parse(s) -> List[List[int]]:
    return [line for group in [parse_line(line.strip()) for line in s.strip().split(";") if line] for line in group if line]


def parse_line(line: str) -> Tuple[List[int], Optional[List[int]]]:
    splits = list(filter(bool,list(map(lambda s: s.strip(), line.split(" ")))))
    first = splits[0]
    OPS = ("ADD", "SHR", "SHL", "NOT", "AND", "OR", "XOR", "CMP")
    OTHER_OPS = ("LD", "ST", "DATA", "JMPR", "JMP")
    last = None
    if first == "DATA" or ('J' in first and "JMPR" not in first):
        last = list(map(int,'{0:010b}'.format(int(splits[-1]))[2:]))
    if first == "IN":
        return (_b("0111") + [0, 0] + _toreg(splits[1])), last
    if first == "OUT":
        return (_b("0111") + [1, 0] + _toreg(splits[1])), last
    if first == "CLF":
        return _b("01100000"), last
    if "J" in first and "JM" not in first:
        return (_b("0101") + [int(s in first) for s in "CAEZ"]), last
    if first in OPS:
        return ([1] + _b(OPS.index(first)) + _rarb(*splits[1:])), last
    if first in OTHER_OPS:
        idx = OTHER_OPS.index(first)
        if idx < 2:
            return ([0] + _b(idx) + _rarb(*splits[1:])), last
        elif idx < 4:
            return ([0] + _b(idx) + [0] * 2 +  _toreg(splits[1])), last
        else:
            return ([0] + _b(idx) + [0] * 4), last
    raise Exception("Should have found something")


def _rarb(a, b):
    return [x for l in (a, b) for x in _toreg(l)]


def _toreg(l):
    l = int(l)
    if l == 0:
        return [0, 0]
    if l == 1:
        return [0, 1]
    if l == 2:
        return [1, 0]
    if l == 3:
        return [1, 1]
    raise Exception("No good")


def _b(s):
    if isinstance(s, int):
        return _b("{0:03b}".format(s))
    return [int(ss) for ss in s]

def unparse(x: int) ->str:
    digits = list(map(int,'{0:010b}'.format(x)))[2:]
    
    is_alu, code, ra, rb = digits[0], digits[1:4], digits[4:6], digits[6:8]
    ra = int(''.join(map(str,ra)), 2)
    rb = int(''.join(map(str,rb)), 2)
    if is_alu:
        op = ("ADD", "SHR", "SHL", "NOT", "AND", "OR", "XOR", "CMP")[int(''.join(map(str,code)), 2)]
        return f'{op} {ra} {rb}'
    else:
        op = ("LD", "ST", "DATA", "JMPR", "JMP", "J", "CLF", "IO")[int(''.join(map(str,code)), 2)]
        if op not in ("DATA", "JMP", "J", "CLF", "IO"):
            return f'{op} {ra} {rb}'
        if op in ("DATA",):
            return f'{op} {rb}'
        if op in ("CLF", "JMP"):
            return op
        if op == "IO":
            direction = "OUT" if digits[4] else "IN"
            return f'{direction} {rb}'
        suffix = ''.join(l for i, l in zip(digits[4:], 'CAEZ') if i)
        return f'J{suffix}'