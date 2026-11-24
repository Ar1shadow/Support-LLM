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
    # 本任务只检查 POINTWISE，PAIRWISE 在 Task 3 改写后再加入
    p = judge.POINTWISE.format(question="q", reference="r", candidate="c")
    assert "{{Customer Support Phone Number}}" in p   # placeholder example survives formatting
    assert "Invented claim" in p and "Invented value" in p
