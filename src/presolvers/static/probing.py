from __future__ import annotations

from src.presolvers.base import StaticPresolvingAlgorithm


class Probing(StaticPresolvingAlgorithm):
    """Placeholder — probing presolver not yet implemented."""

    @property
    def name(self) -> str:
        return "Probing"

    @property
    def slug(self) -> str:
        return "probing_static"

    def _run(self, model) -> None:
        raise NotImplementedError("Probing presolver is not yet implemented.")
