"""Audit runner negative controls; no private or qualification data used."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_agent import git_blob_sha1, validate_source_lock, ReviewIntegrityError, assess, render_markdown


class IndependentReviewAgentTests(unittest.TestCase):
    def fixture(self, root):
        sources = {}
        for i in range(12):
            rel = f"sources/evidence_{i}.txt"
            content = f"source-{i}\n".encode()
            file = root / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
            sources[rel] = content
        lock = {
            "schema": "phm.independent-review.source-lock.v1",
            "target_commit": "a"*40,
            "files": [
                {"path": name, "blob_sha1": git_blob_sha1(value)}
                for name, value in sources.items()
            ],
        }
        return lock, sources

    def test_git_blob_matches_known_git_identity(self):
        self.assertEqual(git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_read_only_lock_accepts_complete_intact_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, _ = self.fixture(root)
            files = validate_source_lock(root, lock, check_git_head=False)
            self.assertEqual(len(files), 12)

    def test_modified_file_fails_closed_not_warn_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, _ = self.fixture(root)
            (root / lock["files"][0]["path"]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ReviewIntegrityError, "SOURCE_BLOB_CHANGED"):
                validate_source_lock(root, lock, check_git_head=False)

    def test_path_traversal_and_duplicate_refuse_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, _ = self.fixture(root)
            lock["files"][0]["path"] = "../private"
            with self.assertRaisesRegex(ReviewIntegrityError, "UNSAFE_OR_DUPLICATE"):
                validate_source_lock(root, lock, check_git_head=False)
            lock, _ = self.fixture(root)
            lock["files"][1] = dict(lock["files"][0])
            with self.assertRaisesRegex(ReviewIntegrityError, "UNSAFE_OR_DUPLICATE"):
                validate_source_lock(root, lock, check_git_head=False)

    def test_missing_file_and_bad_manifest_are_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock, _ = self.fixture(root)
            (root / lock["files"][0]["path"]).unlink()
            with self.assertRaisesRegex(ReviewIntegrityError, "MISSING"):
                validate_source_lock(root, lock, check_git_head=False)
            lock, _ = self.fixture(root)
            lock["files"] = lock["files"][:2]
            with self.assertRaisesRegex(ReviewIntegrityError, "INSUFFICIENT_LOCKED"):
                validate_source_lock(root, lock, check_git_head=False)

    def test_missing_required_review_source_rejects_report(self):
        lock = {"target_commit": "a"*40}
        with self.assertRaisesRegex(ReviewIntegrityError, "REQUIRED_EVIDENCE_NOT_LOCKED"):
            assess(lock, {})

    def test_immutable_markdown_uses_explicit_evidence_status(self):
        report = {
            "reviewed_commit": "a"*40,
            "verdict": "KEEP_DIRECTION_WITH_PIVOT",
            "broad_industrial_qualification": False,
            "reviewer_type": "DETERMINISTIC_READ_ONLY_AGENT_NOT_INDEPENDENT_LLM",
            "findings": [{"priority": "P0", "finding": "example", "source_support": "NOT_VERIFIED",
                          "falsification_test": "independent equipment test"}]
        }
        document = render_markdown(report)
        self.assertIn("NOT_VERIFIED", document)
        self.assertIn("False", document)
        self.assertNotIn("qualification granted", document)


if __name__ == "__main__":
    unittest.main()
