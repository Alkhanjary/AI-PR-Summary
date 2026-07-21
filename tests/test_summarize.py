import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarize_pr import is_ignored, filter_diff, split_by_file, smart_truncate


SAMPLE_DIFF = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
 def hello():
+    print("hi")
     pass
diff --git a/package-lock.json b/package-lock.json
index 3333333..4444444 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,1 +1,1 @@
-old
+new
"""


def test_is_ignored_matches_lockfile():
    assert is_ignored("diff --git a/package-lock.json b/package-lock.json") is True


def test_is_ignored_does_not_match_normal_file():
    assert is_ignored("diff --git a/app.py b/app.py") is False


def test_filter_diff_removes_lockfile_section():
    result = filter_diff(SAMPLE_DIFF)
    assert "app.py" in result
    assert "package-lock.json" not in result


def test_split_by_file_returns_two_chunks():
    chunks = split_by_file(SAMPLE_DIFF)
    assert len(chunks) == 2
    names = [name for name, _ in chunks]
    assert any("app.py" in n for n in names)
    assert any("package-lock.json" in n for n in names)


def test_smart_truncate_keeps_everything_if_under_budget():
    result, omitted = smart_truncate(SAMPLE_DIFF, max_chars=10000)
    assert omitted == []
    assert "app.py" in result


def test_smart_truncate_omits_when_over_budget():
    result, omitted = smart_truncate(SAMPLE_DIFF, max_chars=50)
    assert len(omitted) >= 1
