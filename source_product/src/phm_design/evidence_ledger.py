"""Local evidence-use registry. Audit control, NOT a tamper-proof authority.

Keep one registry for a research program. Deleting it, changing stable lineage
IDs or using another registry can bypass local restrictions. No statistical
independence or sample provenance can be inferred from hashes alone.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from .io import read_json, verify, sha_file


class EvidenceLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS operation (
          key TEXT PRIMARY KEY, kind TEXT NOT NULL, source TEXT NOT NULL,
          output TEXT NOT NULL, status TEXT NOT NULL, manifest TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS lineage_role (
          identity TEXT PRIMARY KEY, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calibration_use (
          domain TEXT NOT NULL, identity TEXT NOT NULL, operation TEXT NOT NULL,
          PRIMARY KEY(domain,identity));
        CREATE TABLE IF NOT EXISTS calibration_numeric_use (
          domain TEXT NOT NULL, signature TEXT NOT NULL, operation TEXT NOT NULL,
          PRIMARY KEY(domain,signature));
        CREATE TABLE IF NOT EXISTS numeric_role (
          domain TEXT NOT NULL, signature TEXT NOT NULL, role TEXT NOT NULL,
          PRIMARY KEY(domain,signature));
        CREATE TABLE IF NOT EXISTS evaluated_source (
          source TEXT PRIMARY KEY, operation TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _identity(self, identity, role):
        previous = self.db.execute("SELECT role FROM lineage_role WHERE identity=?", (identity,)).fetchone()
        if previous and previous[0] != role:
            raise ValueError("LINEAGE_ROLE_REUSE:" + identity)
        self.db.execute("INSERT OR IGNORE INTO lineage_role VALUES (?,?)", (identity, role))

    def claim(self, key, kind, source, output, domain, samples, numeric=()):
        """Reserve before predictions; interrupted/failed exposure is not released.

        ``domain`` may be one device-family string or a collection of equivalent
        representation families participating in one matched evidence operation.
        """
        domains=[domain] if isinstance(domain,str) else list(domain)
        if not domains or any(not isinstance(d,str) or not d for d in domains) or len(set(domains))!=len(domains):
            raise ValueError("EVIDENCE_DOMAIN")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT status,output,manifest FROM operation WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != "COMPLETE":
                    raise ValueError("EVIDENCE_ALREADY_OPENED_REVIEW_REQUIRED")
                cached = Path(old[1])
                verify(cached)
                if sha_file(cached / "manifest.json") != old[2]:
                    raise ValueError("CACHED_EVIDENCE_CHANGED")
                self.db.execute("COMMIT")
                return cached
            if self.db.execute("SELECT 1 FROM evaluated_source WHERE source=?", (source,)).fetchone():
                raise ValueError("SOURCE_HOLDOUT_ALREADY_OPENED")
            for sample in samples:
                role = sample["split"]
                for field in ("group_id", "lineage_id"):
                    self._identity(sample[field], role)
                if kind in {"selection", "risk_calibration"} and role == "calibration":
                    identity = sample["lineage_id"]
                    for one_domain in domains:
                        prior = self.db.execute("SELECT operation FROM calibration_use WHERE domain=? AND identity=?", (one_domain, identity)).fetchone()
                        if prior and prior[0] != key:
                            raise ValueError("CALIBRATION_ALREADY_CONSUMED")
                        self.db.execute("INSERT OR IGNORE INTO calibration_use VALUES (?,?,?)", (one_domain, identity, key))
            for signature, role in numeric:
                for one_domain in domains:
                    old_role = self.db.execute("SELECT role FROM numeric_role WHERE domain=? AND signature=?", (one_domain, signature)).fetchone()
                    if old_role and old_role[0] != role:
                        raise ValueError("NUMERIC_ROLE_REUSE")
                    self.db.execute("INSERT OR IGNORE INTO numeric_role VALUES (?,?,?)", (one_domain, signature, role))
                    if role == "calibration" and kind in {"selection", "risk_calibration"}:
                        old_op = self.db.execute("SELECT operation FROM calibration_numeric_use WHERE domain=? AND signature=?", (one_domain, signature)).fetchone()
                        if old_op and old_op[0] != key:
                            raise ValueError("CALIBRATION_NUMERIC_ALREADY_CONSUMED")
                        self.db.execute("INSERT OR IGNORE INTO calibration_numeric_use VALUES (?,?,?)", (one_domain, signature, key))
            self.db.execute("INSERT INTO operation VALUES (?,?,?,?,?,NULL,NULL)",
                            (key, kind, source, str(Path(output).resolve()), "OPENED"))
            if kind == "evaluation":
                self.db.execute("INSERT INTO evaluated_source VALUES (?,?)", (source, key))
            self.db.execute("COMMIT")
            return None
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def complete(self, key, output):
        verify(Path(output))
        changed = self.db.execute("UPDATE operation SET status='COMPLETE',manifest=? WHERE key=? AND status='OPENED'",
            (sha_file(Path(output) / "manifest.json"), key)).rowcount
        if changed != 1:
            raise ValueError("EVIDENCE_COMMIT_WITHOUT_RESERVATION")

    def fail(self, key, error):
        self.db.execute("UPDATE operation SET status='FAILED',reason=? WHERE key=? AND status='OPENED'", (str(error), key))
