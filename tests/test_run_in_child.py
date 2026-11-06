"""Sync scoring runs in a forked child: results return, and a crashing child only raises."""
import os

import pytest

from orbitol.eval_protocol import run_in_child


def add(a, b, *, scale):
    return {"sum": (a + b) * scale}, [dict(index=0)]


def fail():
    raise ValueError("bad pool")


def abort():
    os.abort()  # stands in for a C extension corrupting the heap


def test_result_and_keywords_return_from_child():
    assert run_in_child(add, 1, 2, scale=3) == ({"sum": 9}, [dict(index=0)])


def test_child_exception_is_reported_with_its_traceback():
    with pytest.raises(RuntimeError, match="ValueError: bad pool"):
        run_in_child(fail)


def test_crashing_child_raises_instead_of_killing_the_parent():
    with pytest.raises(RuntimeError, match="died with exit code -6"):
        run_in_child(abort)
    assert run_in_child(add, 0, 1, scale=1)[0] == {"sum": 1}
