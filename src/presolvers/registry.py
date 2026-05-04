from __future__ import annotations

from src.core.presolving import PresolvingMethod
from src.presolvers.base import PresolvingAlgorithm

# Maps CLI method key → (static class import path, PresolvingMethod enum value)
_METHOD_MAP: dict[str, tuple[str, str, PresolvingMethod]] = {
    "coeff_strengthening": (
        "src.presolvers.static.coefficient_strengthening",
        "CoefficientStrengthening",
        PresolvingMethod.CoeffTightening,
    ),
    "propagation": (
        "src.presolvers.static.constraint_propagation",
        "ConstraintPropagation",
        PresolvingMethod.Propagation,
    ),
    "dual_fix": (
        "src.presolvers.static.dual_fix",
        "DualFix",
        PresolvingMethod.DualFix,
    ),
    "fix_continuous": (
        "src.presolvers.static.fix_continuous",
        "FixContinuous",
        PresolvingMethod.FixContinuous,
    ),
    "sparsify": (
        "src.presolvers.static.sparsify",
        "Sparsify",
        PresolvingMethod.Sparsify,
    ),
    "probing": (
        "src.presolvers.static.probing",
        "Probing",
        PresolvingMethod.Probing,
    ),
}

# Methods only available via PaPILO (no static Python implementation)
_PAPILO_ONLY: dict[str, PresolvingMethod] = {
    "col_singleton": PresolvingMethod.ColSingleton,
    "parallel_cols": PresolvingMethod.ParallelCols,
    "parallel_rows": PresolvingMethod.ParallelRows,
    "simple_probing": PresolvingMethod.SimpleProbing,
    "double_to_neq": PresolvingMethod.DoubleToNEq,
    "simplify_ineq": PresolvingMethod.SimpifyIneq,
    "stuffing": PresolvingMethod.Stuffing,
    "dom_col": PresolvingMethod.DomCol,
    "dual_infer": PresolvingMethod.DualInfer,
    "impl_int": PresolvingMethod.ImplInt,
    "clique_merging": PresolvingMethod.CliqueMerging,
    "substitution": PresolvingMethod.Substitution,
}


def resolve(spec: str) -> PresolvingAlgorithm:
    """
    Parse a method specification and return the corresponding algorithm.

    Grammar:
        "method_key"          → static Python implementation
        "method_key:papilo"   → PaPILO backend restricted to that method
        "method_key:backend"  → raises if backend unknown
        "papilo"              → PaPILO with all default methods

    Examples
    --------
        resolve("coeff_strengthening")        → CoefficientStrengthening()
        resolve("coeff_strengthening:papilo") → PaPILO([PresolvingMethod.CoeffTightening])
        resolve("papilo")                     → PaPILO(methods=None)
    """
    spec = spec.strip()

    if spec == "papilo":
        from src.presolvers.papilo import PaPILO
        return PaPILO(methods=None)

    if ":" in spec:
        method_key, backend = spec.split(":", 1)
        method_key = method_key.strip()
        backend = backend.strip().lower()

        if backend != "papilo":
            raise ValueError(
                f"Unknown backend '{backend}'. Currently supported backends: 'papilo', or omit for static."
            )

        papilo_method = _resolve_papilo_method(method_key)
        from src.presolvers.papilo import PaPILO
        return PaPILO(methods=[papilo_method])

    # No backend specified → use static Python implementation
    if spec not in _METHOD_MAP:
        raise ValueError(
            f"Unknown presolving method '{spec}'. "
            f"Available: {list(_METHOD_MAP.keys())} or 'papilo'."
        )

    module_path, class_name, _ = _METHOD_MAP[spec]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def resolve_many(specs: list[str]) -> list[PresolvingAlgorithm]:
    """Resolve a list of method specs into algorithm instances."""
    return [resolve(s) for s in specs]


def available_methods() -> list[str]:
    """Return all known method keys (static + papilo-only)."""
    return list(_METHOD_MAP.keys()) + list(_PAPILO_ONLY.keys()) + ["papilo"]


def _resolve_papilo_method(method_key: str) -> PresolvingMethod:
    if method_key in _METHOD_MAP:
        return _METHOD_MAP[method_key][2]
    if method_key in _PAPILO_ONLY:
        return _PAPILO_ONLY[method_key]
    raise ValueError(
        f"Unknown method '{method_key}' for PaPILO backend. "
        f"Available: {list(_METHOD_MAP.keys()) + list(_PAPILO_ONLY.keys())}."
    )
