import mip
import numpy as np

from src.core.model import Model
from src.core.presolving import PresolvingMethod
from src.presolvers.algorithm import presolve


def test_coefficient_tightening_reduces_integer_coefficient():
    model = mip.Model()

    x = model.add_var(var_type=mip.BINARY, name="x")
    y = model.add_var(lb=0.0, ub=1.0, name="y")

    model.add_constr(3 * x + 1 * y <= 4)

    wrapper = Model(model=model, path=None)
    changes = presolve(wrapper, method=PresolvingMethod.CoeffTightening)

    assert changes is wrapper

    constr = model.constrs[0]
    coeffs = constr.expr.expr

    # The integer variable x should be removed or tightened to zero
    assert coeffs.get(x, 0.0) == 0.0

    # The remaining continuous variable should remain in the constraint
    assert float(coeffs[y]) == 1.0
    assert float(constr.rhs) == 1.0

    # Model matrix should be updated to reflect the tightened support pattern
    assert wrapper.A.shape == (1, 2)
    assert wrapper.A.nnz == 1


def test_coefficient_tightening_leaves_non_integer_variable_unchanged():
    model = mip.Model()

    x = model.add_var(lb=0.0, ub=2.0, name="x")
    y = model.add_var(lb=0.0, ub=1.0, name="y")

    model.add_constr(2 * x + 1 * y <= 3)

    wrapper = Model(model=model, path=None)
    presolve(wrapper, method=PresolvingMethod.CoeffTightening)

    constr = model.constrs[0]
    coeffs = constr.expr.expr

    # No integer variables are present, so the constraint should remain unchanged
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

    wrapper = Model(model=model, path=None)
    presolve(wrapper, method=PresolvingMethod.CoeffTightening)

    assert len(model.constrs) == 2

    c1 = next(c for c in model.constrs if c.name == "c1")
    c2 = next(c for c in model.constrs if c.name == "c2")

    assert float(c1.expr.expr.get(x, 0.0)) == 0.0
    assert float(c1.expr.expr[z]) == 1.0
    assert float(c1.rhs) == 1.0

    assert float(c2.expr.expr.get(y, 0.0)) == 0.0
    assert float(c2.expr.expr[z]) == 1.0
    assert float(c2.rhs) == 1.0
