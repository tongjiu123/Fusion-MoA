"""One domain-independent read-only Analyst portfolio."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from .task_identity import GENERAL_TASK_TYPE


@dataclass(frozen=True)
class AnalystRoleSpec:
    role: str
    actor_hint: str
    focus: str


@dataclass(frozen=True)
class AnalystPortfolio:
    portfolio_id: str
    supported_task_types: frozenset[str]
    roles: Tuple[AnalystRoleSpec, ...]

    def supports(self, task_type: str) -> bool:
        return task_type in self.supported_task_types


@dataclass(frozen=True)
class PortfolioSelection:
    task_type: str
    status: str
    portfolio: Optional[AnalystPortfolio]
    reason: str
    fallback_used: bool = False

    @property
    def portfolio_id(self) -> Optional[str]:
        return self.portfolio.portfolio_id if self.portfolio else None

    @property
    def wrong_domain(self) -> bool:
        return bool(
            self.portfolio is not None
            and not self.portfolio.supports(self.task_type)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "status": self.status,
            "portfolio_id": self.portfolio_id,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "wrong_domain": self.wrong_domain,
            "analyst_roles": (
                [role.role for role in self.portfolio.roles]
                if self.portfolio
                else []
            ),
        }


GENERIC_EVIDENCE_PORTFOLIO = AnalystPortfolio(
    portfolio_id="generic_evidence_v1",
    supported_task_types=frozenset({GENERAL_TASK_TYPE}),
    roles=(
        AnalystRoleSpec(
            role="alternative_hypothesis",
            actor_hint="D7",
            focus=(
                "Independently find the smallest alternative explanation "
                "supported by the current conversation and observations. "
                "State one falsifiable probe without assuming a domain, "
                "application, or evaluation method."
            ),
        ),
        AnalystRoleSpec(
            role="requirement_evidence_auditor",
            actor_hint="D9",
            focus=(
                "Independently audit whether the available evidence actually "
                "supports the user's observable request. Identify one "
                "unsupported assumption and a falsifiable observation."
            ),
        ),
        AnalystRoleSpec(
            role="context_assumption_auditor",
            actor_hint="D13",
            focus=(
                "Independently inspect context, state, interface, and "
                "interpretation assumptions using only the frozen evidence. "
                "Provide one falsifiable probe and no executable instruction."
            ),
        ),
    ),
)


def select_generic_portfolio(
    actor_hints: Sequence[str] = ("D7", "D9", "D13"),
) -> PortfolioSelection:
    actors = tuple(actor_hints)
    if len(actors) != len(GENERIC_EVIDENCE_PORTFOLIO.roles):
        raise ValueError("generic portfolio requires exactly three Analyst cells")
    portfolio = AnalystPortfolio(
        portfolio_id=GENERIC_EVIDENCE_PORTFOLIO.portfolio_id,
        supported_task_types=GENERIC_EVIDENCE_PORTFOLIO.supported_task_types,
        roles=tuple(
            replace(role, actor_hint=actor)
            for role, actor in zip(
                GENERIC_EVIDENCE_PORTFOLIO.roles,
                actors,
            )
        ),
    )
    return PortfolioSelection(
        task_type=GENERAL_TASK_TYPE,
        status="selected",
        portfolio=portfolio,
        reason="domain_independent_generic_portfolio",
    )


__all__ = [
    "AnalystPortfolio",
    "AnalystRoleSpec",
    "GENERIC_EVIDENCE_PORTFOLIO",
    "PortfolioSelection",
    "select_generic_portfolio",
]
