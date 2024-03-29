import parser
from pprint import pprint

import pytest


def test_parser():
    txt = """
ADD   1 2;
ST 0 2;
SHL   3 0;
LD 1   3;
DATA 2 5  ;     # asdfasd 
JMPR   1;    # asdfasd 
JMP   6;  
CLF   ;
JC 7;
JCA 8;
JCEZ 9;
J 10;
IN 3;
OUT 1;"""
    parsed = parser.parse(txt)
    assert parsed == [
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 1],
    ]

def test_unparse():
    expected = """
ADD 1 2;
ST 0 2;
SHL 3 0;
LD 1 3;
DATA 2;
JMPR 0 1;
JMP;
CLF;
JC;
JCA;
JCEZ;
J;
IN 3;
OUT 1;"""
    parsed = [
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 1],
    ]
    unparsed = [
        parser.unparse(int(''.join(map(str,line)), 2))
        for line in parsed
    ]
    assert unparsed == [
        line.strip()
        for line in expected.split(';')
        if line
    ]