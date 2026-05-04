import mip
import numpy as np

from src.core.model import Model
from src.presolvers.static.coefficient_strengthening import CoefficientStrengthening


def _run(mip_model: mip.Model) -> CoefficientStrengthening:
    wrapper = Model(model=mip_model, path=None)
    algo = CoefficientStrengthening()
    algo._run(wrapper)
    return algo


def test_coefficient_tightening_reduces_integer_coefficient():
    model = mip.Model()

    x = model.add_var(var_type=mip.BINARY, name="x")
    y = model.add_var(lb=0.0, ub=1.0, name="y")

    model.add_constr(3 * x + 1 * y <= 4)

    _run(model)

    constr = model.constrs[0]
    coeffs = constr.expr.expr

    assert coeffs.get(x, 0.0) == 0.0
    assert float(coeffs[y]) == 1.0
    assert float(constr.rhs) == 1.0


def test_coefficient_tightening_leaves_non_integer_variable_unchanged():
    model = mip.Model()

    x = model.add_var(lb=0.0, ub=2.0, name="x")
    y = model.add_var(lb=0.0, ub=1.0, name="y")

    model.add_constr(2 * x + 1 * y <= 3)

    _run(model)

    constr = model.constrs[0]
    coeffs = constr.expr.expr

    assert float(coeffs[x]) == 2.0
    assert float(coeffs[y]) == 1.0
    assert float(constr.rhs) == 3.0


def test_coefficient_tightening_applies_to_multiple_constraints_in_batch():
    model = mip.Model()

    x = model.add_var(var_type=mip.BINARY, name="x")
    y = model.add_var(var_type=mip.BINARY, name="y")
    z = model.add_var(lb=0.0, ub=1.0, name="z")

    model.add_constr(3 * x + 1 * z <= 4, name="c1")
    model.add_constr(2 * y + 1 * z <= 3, name="c2")

    _run(model)

    assert len(model.constrs) == 2

    c1 = next(c for c in model.constrs if c.name == "c1")
    c2 = next(c for c in model.constrs if c.name == "c2")

    assert float(c1.expr.expr.get(x, 0.0)) == 0.0
    assert float(c1.expr.expr[z]) == 1.0
    assert float(c1.rhs) == 1.0

    assert float(c2.expr.expr.get(y, 0.0)) == 0.0
    assert float(c2.expr.expr[z]) == 1.0
    assert float(c2.rhs) == 1.0
