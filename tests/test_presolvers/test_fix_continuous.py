import mip
import pytest

from src.core.model import Model
from src.presolvers.techniques.fix_continuous import FixContinuous


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fc(model: mip.Model) -> FixContinuous:
    wrapper = Model(model=model, path=None)
    return FixContinuous(wrapper)


# ---------------------------------------------------------------------------
# Basic fixing
# ---------------------------------------------------------------------------

def test_fixed_variable_is_removed():
    """A continuous variable with lb == ub should be removed from the model."""
    m = mip.Model()
    x = m.add_var(lb=2.0, ub=2.0, name="x")  # fixed at 2
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 10)

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 1
    assert len(m.vars) == 1
    assert m.vars[0].name == "y"


def test_fixed_variable_shifts_rhs():
    """Substituting x=2 into x + y <= 10 should give y <= 8."""
    m = mip.Model()
    x = m.add_var(lb=2.0, ub=2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 10, name="c1")

    fc = _make_fc(m)
    fc.presolve()

    constr = m.constrs[0]
    assert float(constr.rhs) == pytest.approx(8.0)


def test_fixed_variable_with_negative_coefficient():
    """x=3, coefficient -2: rhs shifts by +6.  -2x + y <= 10 => y <= 16."""
    m = mip.Model()
    x = m.add_var(lb=3.0, ub=3.0, name="x")
    y = m.add_var(lb=0.0, ub=20.0, name="y")
    m.add_constr(-2 * x + y <= 10, name="c1")

    fc = _make_fc(m)
    fc.presolve()

    constr = m.constrs[0]
    assert float(constr.rhs) == pytest.approx(16.0)


def test_fixed_variable_in_equality_constraint():
    """x=1 in x + y == 5 should give y == 4."""
    m = mip.Model()
    x = m.add_var(lb=1.0, ub=1.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, name="y")
    m.add_constr(x + y == 5, name="eq")

    fc = _make_fc(m)
    fc.presolve()

    constr = m.constrs[0]
    assert float(constr.rhs) == pytest.approx(4.0)


def test_fixed_variable_in_ge_constraint():
    """x=2 in x + y >= 6 should give y >= 4."""
    m = mip.Model()
    x = m.add_var(lb=2.0, ub=2.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, name="y")
    m.add_constr(x + y >= 6, name="ge")

    fc = _make_fc(m)
    fc.presolve()

    constr = m.constrs[0]
    assert float(constr.rhs) == pytest.approx(4.0)


def test_fixed_variable_appears_in_multiple_constraints():
    """x fixed at 3; both constraints should have their RHS updated."""
    m = mip.Model()
    x = m.add_var(lb=3.0, ub=3.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, name="y")
    z = m.add_var(lb=0.0, ub=10.0, name="z")
    m.add_constr(x + y <= 10, name="c1")   # => y <= 7
    m.add_constr(x + z <= 8, name="c2")    # => z <= 5

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 1
    rhs_map = {c.name: float(c.rhs) for c in m.constrs}
    assert rhs_map["c1"] == pytest.approx(7.0)
    assert rhs_map["c2"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# No fixing cases
# ---------------------------------------------------------------------------

def test_no_fix_when_bounds_differ():
    """Variables with lb != ub must not be removed."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=5.0, name="x")
    y = m.add_var(lb=0.0, ub=3.0, name="y")
    m.add_constr(x + y <= 6)

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 0
    assert len(m.vars) == 2


def test_no_fix_for_integer_variable():
    """Integer variables with lb == ub should not be fixed by FixContinuous."""
    m = mip.Model()
    x = m.add_var(var_type=mip.INTEGER, lb=2.0, ub=2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 8)

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 0
    assert len(m.vars) == 2


def test_no_fix_for_binary_variable():
    """Binary variables with lb == ub should not be fixed by FixContinuous."""
    m = mip.Model()
    b = m.add_var(var_type=mip.BINARY, name="b")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(b + y <= 5)

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 0


# ---------------------------------------------------------------------------
# Multiple fixed variables
# ---------------------------------------------------------------------------

def test_multiple_fixed_variables_all_removed():
    """Two fixed continuous variables should both be removed in one pass."""
    m = mip.Model()
    x = m.add_var(lb=1.0, ub=1.0, name="x")
    y = m.add_var(lb=2.0, ub=2.0, name="y")
    z = m.add_var(lb=0.0, ub=10.0, name="z")
    m.add_constr(x + y + z <= 10, name="c1")  # => z <= 7

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 2
    assert len(m.vars) == 1
    assert m.vars[0].name == "z"
    assert float(m.constrs[0].rhs) == pytest.approx(7.0)


def test_fixed_at_zero_removes_variable_no_rhs_shift():
    """A variable fixed at 0 is removed but the RHS is unchanged."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=0.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 4, name="c1")

    fc = _make_fc(m)
    changes = fc.presolve()

    assert changes == 1
    assert len(m.vars) == 1
    assert float(m.constrs[0].rhs) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Return value / matrix update
# ---------------------------------------------------------------------------

def test_presolve_returns_count_of_fixed_variables():
    m = mip.Model()
    x = m.add_var(lb=3.0, ub=3.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 8)

    fc = _make_fc(m)
    result = fc.presolve()

    assert isinstance(result, int)
    assert result == 1


def test_matrix_updated_after_fixing():
    """After fixing, update_matrix should reflect the reduced constraint."""
    m = mip.Model()
    x = m.add_var(lb=2.0, ub=2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, name="y")
    m.add_constr(x + y <= 8)

    wrapper = Model(model=m, path=None)
    fc = FixContinuous(wrapper)
    fc.presolve()

    # One constraint, one remaining variable
    assert wrapper.A.shape == (1, 1)
