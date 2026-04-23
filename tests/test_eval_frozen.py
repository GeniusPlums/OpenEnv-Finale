import hashlib
import pathlib


def test_eval_frozen():
    data = pathlib.Path("data/scenarios/eval.jsonl").read_bytes()
    expected = pathlib.Path("data/scenarios/eval.jsonl.lock").read_text().strip()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected, (
        "eval.jsonl changed after freeze. Either revert your edits, or "
        "(ONLY if you intentionally changed the eval set) regenerate the "
        ".lock file and acknowledge this breaks eval comparability with prior runs."
    )
