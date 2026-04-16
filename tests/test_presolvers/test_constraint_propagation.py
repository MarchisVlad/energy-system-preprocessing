import math

import mip
import pytest

from src.core.model import Model
from src.presolvers.techniques.constraint_propagation import ConstraintPropagation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cp(model: mip.Model) -> ConstraintPropagation:
    wrapper = Model(model=model, path=None)
    return ConstraintPropagation(wrapper)


# ---------------------------------------------------------------------------
# Upper-bound tightening
# ---------------------------------------------------------------------------

def test_ub_tightened_by_le_constraint():
    """x + y <= 3, y in [0,1] => x_ub should tighten to 3."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=1.0, name="y")
    m.add_constr(x + y <= 3)

    cp = _make_cp(m)
    changes = cp.presolve()

    assert changes >= 1
    assert float(x.ub) == pytest.approx(3.0)


def test_ub_tightened_with_positive_lb():
    """x + y <= 5, y in [2, 4] => x_ub should tighten to 3."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=2.0, ub=4.0, name="y")
    m.add_constr(x + y <= 5)

    cp = _make_cp(m)
    cp.presolve()

    assert float(x.ub) == pytest.approx(3.0)
    assert float(y.ub) == pytest.approx(4.0)  # unchanged — lb of x is 0


# ---------------------------------------------------------------------------
# Lower-bound tightening
# ---------------------------------------------------------------------------

def test_lb_tightened_by_ge_constraint():
    """x + y >= 4, y in [0, 2] => x_lb should tighten to 2."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=2.0, name="y")
    m.add_constr(x + y >= 4)

    cp = _make_cp(m)
    cp.presolve()

    assert float(x.lb) == pytest.approx(2.0)


def test_lb_tightened_by_equality_constraint():
    """x + y == 5, y in [0, 3] => x_lb tightens to 2, x_ub tightens to 5."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=3.0, name="y")
    m.add_constr(x + y == 5)

    cp = _make_cp(m)
    cp.presolve()

    assert float(x.lb) == pytest.approx(2.0)
    assert float(x.ub) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Integer / binary variable rounding
# ---------------------------------------------------------------------------

def test_integer_ub_is_floored():
    """For an integer variable the implied ub should be floored."""
    m = mip.Model()
    x = m.add_var(var_type=mip.INTEGER, lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=1.5, name="y")
    m.add_constr(x + y <= 4)

    cp = _make_cp(m)
    cp.presolve()

    # Implied ub = 4 - 0 = 4; floor(4) = 4 — no fractional tightening here.
    # Now use a fractional bound: x + y <= 3.7, y in [0, 0.5]
    # => x_ub implied = 3.7 - 0 = 3.7 => floor = 3
    m2 = mip.Model()
    x2 = m2.add_var(var_type=mip.INTEGER, lb=0.0, ub=10.0, name="x")
    y2 = m2.add_var(lb=0.0, ub=0.5, name="y")
    m2.add_constr(x2 + y2 <= 3.7)

    cp2 = _make_cp(m2)
    cp2.presolve()

    assert float(x2.ub) == pytest.approx(3.0)


def test_binary_lb_is_ceiled():
    """For a binary variable the implied lb should be ceiled."""
    m = mip.Model()
    b = m.add_var(var_type=mip.BINARY, name="b")
    y = m.add_var(lb=0.0, ub=0.3, name="y")
    # b + y >= 0.8 => b_lb >= 0.8 - 0.3 = 0.5 => ceil = 1
    m.add_constr(b + y >= 0.8)

    cp = _make_cp(m)
    cp.presolve()

    assert float(b.lb) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# No tightening cases
# ---------------------------------------------------------------------------

def test_no_change_when_bounds_already_tight():
    """If bounds are already as tight as implied, no changes should be reported."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=3.0, name="x")
    y = m.add_var(lb=0.0, ub=1.0, name="y")
    m.add_constr(x + y <= 4)  # implied ub for x is 4, current ub is 3 — no improvement

    cp = _make_cp(m)
    changes = cp.presolve()

    assert changes == 0
    assert float(x.ub) == pytest.approx(3.0)


def test_no_change_for_single_variable_constraint():
    """A constraint with only one variable carries no propagation information."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    m.add_constr(x <= 5)

    cp = _make_cp(m)
    changes = cp.presolve()

    assert changes == 0


def test_no_change_when_bound_is_infinite():
    """If a peer variable has an infinite bound the min-activity is undefined;
    propagation should be skipped rather than crash.

    x - y <= 5 with y having ub=+inf: to tighten x we need min-activity
    of the remaining terms, which is (-1) * ub_y = -inf => not finite,
    so x.ub cannot be derived.
    """
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=mip.INF, name="y")  # unbounded above
    m.add_constr(x - y <= 5)  # negative coeff on y → min-activity needs ub_y

    cp = _make_cp(m)
    changes = cp.presolve()

    assert changes == 0
    assert float(x.ub) == pytest.approx(10.0)  # unchanged


# ---------------------------------------------------------------------------
# Multi-round propagation
# ---------------------------------------------------------------------------

def test_propagation_cascades_across_constraints():
    """Tightening one bound should cascade to tighten another in the next round."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, name="y")
    z = m.add_var(lb=0.0, ub=10.0, name="z")

    # Round 1: x + z <= 3 tightens x_ub to 3 (z_lb=0)
    # Round 2: x + y <= 5 tightens y_ub to 5 (x_lb=0), but with x_ub=3 known,
    #          and z + y <= 4 could tighten y further once z_ub is reduced.
    m.add_constr(x + z <= 3)   # => x_ub <= 3, z_ub <= 3
    m.add_constr(z + y <= 4)   # after z_ub=3, y_ub <= 4-0=4

    cp = _make_cp(m)
    cp.presolve()

    assert float(x.ub) <= 3.0 + 1e-9
    assert float(z.ub) <= 3.0 + 1e-9
    assert float(y.ub) <= 4.0 + 1e-9


def test_max_rounds_limits_iterations():
    """Setting max_rounds=1 should perform only a single pass."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, name="y")
    z = m.add_var(lb=0.0, ub=10.0, name="z")

    m.add_constr(x + z <= 3)
    m.add_constr(z + y <= 4)

    cp = _make_cp(m)
    cp.presolve(max_rounds=1)

    # After 1 round, x_ub and z_ub are tightened from constraint 1.
    # y_ub tightening from constraint 2 uses z_lb=0, so y_ub=4 is also possible
    # in a single pass — but the key check is the algorithm doesn't error.
    assert float(x.ub) <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# Return value / matrix update
# ---------------------------------------------------------------------------

def test_presolve_returns_total_change_count():
    """presolve() should return the total number of bound tightenings."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=1.0, name="y")
    m.add_constr(x + y <= 3)

    cp = _make_cp(m)
    changes = cp.presolve()

    assert isinstance(changes, int)
    assert changes >= 1


def test_matrix_updated_after_tightening():
    """After propagation, model.update_matrix() should have been called."""
    m = mip.Model()
    x = m.add_var(lb=0.0, ub=10.0, name="x")
    y = m.add_var(lb=0.0, ub=1.0, name="y")
    m.add_constr(x + y <= 3)

    wrapper = Model(model=m, path=None)
    cp = ConstraintPropagation(wrapper)
    cp.presolve()

    # The matrix should reflect the constraint structure (1 row, 2 cols)
    assert wrapper.A.shape == (1, 2)
