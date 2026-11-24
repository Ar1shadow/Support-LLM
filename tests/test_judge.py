import os
os.environ.setdefault("JUDGE_API_KEY", "x"); os.environ.setdefault("JUDGE_MODEL", "x")
import judge

def test_parse_invented_valid():
    raw = [{"text": "3-5 business days", "kind": "value"}, {"text": "a {{Fee}} applies", "kind": "claim"}]
    assert judge.parse_invented(raw) == raw

def test_parse_invented_malformed():
    assert judge.parse_invented(None) == []
    assert judge.parse_invented("none") == []
    assert judge.parse_invented([{"text": "x", "kind": "other"}, {"kind": "value"}, "y"]) == []

def test_kinds():
    assert judge.kinds([{"text": "a", "kind": "claim"}, {"text": "b", "kind": "claim"}]) == {"claim"}

def test_prompts_format():
    p = judge.POINTWISE.format(question="q", reference="r", candidate="c")
    assert "{{Customer Support Phone Number}}" in p   # placeholder example survives formatting
    assert "Invented claim" in p and "Invented value" in p

def test_pairwise_format():
    p = judge.PAIRWISE.format(question="q", a="aa", b="bb")
    assert "{{Customer Support Phone Number}}" in p
    assert "Invented claim" in p and "Invented value" in p
    assert '"invented_a"' in p and '"invented_b"' in p

def test_merge_invented_swaps_order():
    v = {"text": "3-5 days", "kind": "value"}; c = {"text": "payment plans", "kind": "claim"}
    r1 = {"invented_a": [v], "invented_b": []}
    r2 = {"invented_a": [c], "invented_b": []}      # swapped: slot A == response b
    a, b = judge.merge_invented(r1, r2)
    assert judge.kinds(a) == {"value"} and judge.kinds(b) == {"claim"}

def test_merge_invented_missing():
    assert judge.merge_invented({}, {}) == ([], [])

def test_merge_invented_malformed_and_dedup():
    v = {"text": "3-5 days", "kind": "value"}
    r1 = {"invented_a": [v], "invented_b": "none"}
    r2 = {"invented_a": 5, "invented_b": [v]}       # swapped: invented_b == a
    a, b = judge.merge_invented(r1, r2)
    assert a == [v] and b == []

def test_compare_keeps_invented_when_one_order_fails(monkeypatch):
    v = {"text": "x", "kind": "value"}
    outs = iter([{"winner": "A", "invented_a": [v], "invented_b": []}, None])
    monkeypatch.setattr(judge, "_call", lambda prompt: next(outs))
    r = judge.compare("q", "a", "b")
    assert r["score"] is None and r["invented_a"] == [v] and r["invented_b"] == []
