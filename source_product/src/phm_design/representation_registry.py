"""Capability-driven representation registry for cross-domain automatic design.

This module does not choose a winner from performance. It answers a narrower,
pre-test question: which representation pipelines are legally available for a
(domain, task, artifact-capability) tuple. This keeps domain support explicit
and auditable while allowing the designer/orchestrator to enumerate legal arms
without hard-coding Thermal-only assumptions in study scripts.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

DOMAINS = ("eps", "thermal", "data")
TASKS = ("multicomponent", "family")

@dataclass(frozen=True)
class RepresentationCandidate:
    id: str
    domain: str
    task: str
    source: str
    version: str
    requires: tuple[str, ...] = ()
    research_only: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["requires"] = list(self.requires)
        return d


def _validate(domain: str, task: str) -> None:
    if domain not in DOMAINS:
        raise ValueError("UNKNOWN_DOMAIN")
    if task not in TASKS:
        raise ValueError("UNKNOWN_REPRESENTATION_TASK")


def representation_candidates(
    domain: str,
    *,
    task: str = "multicomponent",
    available_artifacts: Iterable[str] = (),
) -> list[RepresentationCandidate]:
    """Return only representation arms legal before model/selection/test use.

    `available_artifacts` expresses already-frozen prerequisites such as an
    independent commissioning profile. The registry never infers that such an
    artifact exists and never generates it implicitly.
    """
    _validate(domain, task)
    have = frozenset(str(x) for x in available_artifacts)
    out: list[RepresentationCandidate] = []
    if task == "multicomponent":
        out.extend([
            RepresentationCandidate("raw", domain, task, "sat_multicomponent", "raw.v1"),
            RepresentationCandidate("mechanism.v1", domain, task, "multicomponent_features", "v1"),
        ])
        if domain == "thermal":
            out.append(RepresentationCandidate("mechanism.v2", domain, task, "multicomponent_features", "v2"))
        return out

    # Family pipeline is the cross-domain passport/context/mechanism path.
    out.extend([
        RepresentationCandidate("family.context.v1", domain, task, "family_pipeline", "context.v1"),
        RepresentationCandidate("family.mechanism.v1", domain, task, "family_pipeline", "mechanism.v1"),
    ])
    if domain == "thermal":
        out.append(RepresentationCandidate("family.thermal.rolling.v2", domain, task, "thermal_family_v2", "v2"))
    if "commissioning_profile" in have:
        out.append(RepresentationCandidate(
            "family.commissioned.v1", domain, task, "commissioning", "commissioned.v1",
            ("commissioning_profile",),
        ))
    if {"commissioning_profile", "hybrid_profile"}.issubset(have):
        out.append(RepresentationCandidate(
            "family.hybrid.v1", domain, task, "commissioning_hybrid", "hybrid.v1",
            ("commissioning_profile", "hybrid_profile"),
        ))
    return out



@dataclass(frozen=True)
class CoDesignArmPlan:
    id: str
    domain: str
    task: str
    representation_id: str
    prepare_command: str | None
    prepare_mode: str
    architecture_command: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["requires"] = list(self.requires)
        return d


def codesign_arm_plan(
    domain: str,
    *,
    task: str = "multicomponent",
    available_artifacts: Iterable[str] = (),
) -> list[CoDesignArmPlan]:
    """Map legal representation candidates to executable preparation/design entrypoints.

    This function is deliberately pre-performance: it never reads metrics, selection
    artifacts, risk data or test data. It only translates capability legality into
    auditable execution arms that reuse existing CLI paths.
    """
    reps = representation_candidates(
        domain, task=task, available_artifacts=available_artifacts
    )
    out: list[CoDesignArmPlan] = []
    for rep in reps:
        if task == "multicomponent":
            if rep.id == "raw":
                prepare_command = None
                prepare_mode = "use_native_dataset_directly"
            elif rep.id == "mechanism.v1":
                prepare_command = "prepare-multicomponent --representation-version v1"
                prepare_mode = "prepared_dataset"
            elif rep.id == "mechanism.v2":
                prepare_command = "prepare-multicomponent --representation-version v2"
                prepare_mode = "prepared_dataset"
            else:  # pragma: no cover - registry and mapper must evolve together
                raise ValueError("UNMAPPED_MULTICOMPONENT_REPRESENTATION")
            architecture_command = "design-reference"
        else:
            if rep.id == "family.context.v1":
                prepare_command = "prepare-family --mode context"
                prepare_mode = "prepared_dataset"
            elif rep.id == "family.mechanism.v1":
                prepare_command = "prepare-family --mode mechanism"
                prepare_mode = "prepared_dataset"
            elif rep.id == "family.thermal.rolling.v2":
                prepare_command = "prepare-family-thermal-v2"
                prepare_mode = "prepared_dataset"
            elif rep.id == "family.commissioned.v1":
                prepare_command = "prepare-commissioned-family"
                prepare_mode = "prepared_dataset"
            elif rep.id == "family.hybrid.v1":
                prepare_command = "prepare-hybrid"
                prepare_mode = "prepared_dataset"
            else:  # pragma: no cover
                raise ValueError("UNMAPPED_FAMILY_REPRESENTATION")
            architecture_command = "design-family"
        out.append(CoDesignArmPlan(
            id=f"{domain}:{rep.id}",
            domain=domain,
            task=task,
            representation_id=rep.id,
            prepare_command=prepare_command,
            prepare_mode=prepare_mode,
            architecture_command=architecture_command,
            requires=rep.requires,
        ))
    return out


def multidomain_codesign_plan(
    *,
    task: str = "multicomponent",
    available_artifacts_by_domain: dict[str, Iterable[str]] | None = None,
) -> dict:
    available_artifacts_by_domain = available_artifacts_by_domain or {}
    arms = {
        d: [x.to_dict() for x in codesign_arm_plan(
            d, task=task,
            available_artifacts=available_artifacts_by_domain.get(d, ()),
        )]
        for d in DOMAINS
    }
    return {
        "schema": "phm.representation-architecture-codesign-plan.v1",
        "task": task,
        "domains": list(DOMAINS),
        "arms": arms,
        "representation_winner_selected": False,
        "architecture_winner_selected": False,
        "risk_used": False,
        "test_used": False,
        "production_qualified": False,
    }

def multidomain_representation_plan(
    *,
    task: str = "multicomponent",
    available_artifacts_by_domain: dict[str, Iterable[str]] | None = None,
) -> dict:
    available_artifacts_by_domain = available_artifacts_by_domain or {}
    arms = {
        d: [x.to_dict() for x in representation_candidates(
            d, task=task, available_artifacts=available_artifacts_by_domain.get(d, ())
        )]
        for d in DOMAINS
    }
    return {
        "schema": "phm.representation-capability-plan.v1",
        "task": task,
        "domains": list(DOMAINS),
        "arms": arms,
        "winner_selected": False,
        "test_used": False,
        "production_qualified": False,
    }
