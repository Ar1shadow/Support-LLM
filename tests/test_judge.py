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
    w = judge.PAIRWISE.format(question="q", a="aa", b="bb")
    for t in (p, w):
        assert "{{Order Number}}" in t            # 四层花括号渲染成占位符
        assert "Invented claim" in t and "Invented value" in t
        assert "never an invented fact" in t
        assert "does not make it acceptable" not in t   # 旧规则：占位符包装也算 claim，已删除
        assert '"kind": "value" | "claim"' in t
    assert '"invented_a"' in w and '"invented_b"' in w

def test_invented_rules_stay_short():
    assert len(judge.INVENTED_RULES) < 1100

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

def test_score_kept_when_invented_malformed(monkeypatch):
    monkeypatch.setattr(judge, "_call", lambda p: {"score": 7, "reason": "r", "invented": "none"})
    assert judge.score_with_reason("q", "r", "c") == (7, "r", [])

def test_score_non_integer(monkeypatch):
    monkeypatch.setattr(judge, "_call", lambda p: {"score": "7/10"})
    assert judge.score_with_reason("q", "r", "c") == (None, None, [])
    monkeypatch.setattr(judge, "_call", lambda p: {"score": None})
    assert judge.score_with_reason("q", "r", "c") == (None, None, [])

def test_merge_invented_r1_failed():
    c = {"text": "plans", "kind": "claim"}; v = {"text": "5 days", "kind": "value"}
    a, b = judge.merge_invented({}, {"invented_a": [c], "invented_b": [v]})
    assert a == [v] and b == [c]

def test_parse_invented_normalises_kind():
    assert judge.parse_invented([{"text": "x", "kind": " Claim "}]) == [{"text": "x", "kind": "claim"}]

def test_parse_invented_warns_on_drop(capsys):
    judge.parse_invented([{"text": "x", "kind": "other"}, "y"])
    assert capsys.readouterr().out.count("judge warning: dropped invented entry") == 2
    judge.parse_invented(None)
    assert capsys.readouterr().out == ""

def test_extract_json():
    c = 'Sure: {"score": 5, "invented": [{"text": "{{X}}", "kind": "claim"}], "reason": "a {b}"} trailing }'
    assert judge._extract_json(c)["score"] == 5
    assert judge._extract_json('{"x": 1} then {"winner": "A"}') == {"winner": "A"}
    assert judge._extract_json("no json {oops") is None

def test_new_prompt_sentences():
    p = judge.POINTWISE.format(question="q", reference="r", candidate="c")
    w = judge.PAIRWISE.format(question="q", a="a", b="b")
    for t in (p, w):
        assert "Hedging" in t and "does not excuse it" in t
    assert "may itself contain invented items" in p
    assert "if both invent, prefer" in w
