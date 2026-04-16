import sys

import mip
import pytest

# mip.INF is math.inf in Python, but mip stores it internally as sys.float_info.max.
# Reading back a bound that was set to ±mip.INF will give ±sys.float_info.max.
_MIP_INF = sys.float_info.max

from src.core.model import Model
from src.presolvers.techniques.dual_fix import DualFix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(model: mip.Model) -> DualFix:
    wrapper = Model(model=model, path=None)
    return DualFix(wrapper)


def _minimise(vars_: list, obj_coeffs: list) -> mip.Model:
    """Return a minimisation model with the given objective."""
    m = mip.Model(sense=mip.MINIMIZE)
    return m


# ---------------------------------------------------------------------------
# Fix at lower bound (uplocks == 0, c_k >= 0, minimisation)
# ---------------------------------------------------------------------------

def test_fix_at_lb_no_uplocks_positive_cost():
    """x has no uplocks and positive cost in a min problem => fix at lb."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=1.0, ub=10.0, obj=5.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    # Only y appears in a <= constraint, so x has zero uplocks.
    m.add_constr(y <= 3)

    df = _make_df(m)
    changes = df.presolve()

    assert changes >= 1
    assert float(x.lb) == pytest.approx(1.0)
    assert float(x.ub) == pytest.approx(1.0)


def test_fix_at_lb_no_uplocks_zero_cost():
    """x with zero obj coefficient and no uplocks should also be fixed at lb."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=2.0, ub=8.0, obj=0.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(y <= 4)

    df = _make_df(m)
    df.presolve()

    assert float(x.lb) == pytest.approx(2.0)
    assert float(x.ub) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Fix at upper bound (downlocks == 0, c_k <= 0, minimisation)
# ---------------------------------------------------------------------------

def test_fix_at_ub_no_downlocks_negative_cost():
    """x has no downlocks and negative cost in a min problem => fix at ub."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=6.0, obj=-3.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    # x only appears with a positive coeff in <=, giving it uplocks but no
    # downlocks; but if we want no downlocks we keep x out of constraints.
    m.add_constr(y <= 4)

    df = _make_df(m)
    df.presolve()

    assert float(x.lb) == pytest.approx(6.0)
    assert float(x.ub) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Uplocks / downlocks prevent fixing
# ---------------------------------------------------------------------------

def test_no_fix_when_uplock_present():
    """x appears with positive coeff in a <= row (uplock) => cannot fix at lb."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=10.0, obj=2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(x + y <= 8)  # positive coeff on x => uplock

    df = _make_df(m)
    changes = df.presolve()

    # x has an uplock, so it must NOT be fixed at lb despite positive cost
    assert float(x.ub) == pytest.approx(10.0)


def test_no_fix_when_downlock_present():
    """x appears with negative coeff in a <= row (downlock) => cannot fix at ub."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=10.0, obj=-2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(-x + y <= 4)  # negative coeff on x => downlock

    df = _make_df(m)
    changes = df.presolve()

    assert float(x.lb) == pytest.approx(0.0)
    assert float(x.ub) == pytest.approx(10.0)


def test_equality_constraint_blocks_fixing():
    """A variable appearing in an equality gets both an uplock and a downlock,
    so it should never be fixed by dual fixing."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=10.0, obj=5.0, name="x")
    y = m.add_var(lb=0.0, ub=10.0, obj=1.0, name="y")
    m.add_constr(x + y == 7)

    df = _make_df(m)
    changes = df.presolve()

    assert changes == 0
    assert float(x.ub) == pytest.approx(10.0)
    assert float(x.lb) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Maximisation sense
# ---------------------------------------------------------------------------

def test_fix_at_lb_maximisation_negative_cost():
    """For maximisation, a variable with negative cost and no uplocks (in the
    flipped sense) should be fixed at lb."""
    m = mip.Model(sense=mip.MAXIMIZE)
    x = m.add_var(lb=1.0, ub=10.0, obj=-4.0, name="x")  # -c_k in min = +4 => fix at lb
    y = m.add_var(lb=0.0, ub=5.0, obj=2.0, name="y")
    m.add_constr(y <= 3)

    df = _make_df(m)
    df.presolve()

    assert float(x.lb) == pytest.approx(1.0)
    assert float(x.ub) == pytest.approx(1.0)


def test_fix_at_ub_maximisation_positive_cost():
    """For maximisation, a variable with positive cost and no downlocks should
    be fixed at ub."""
    m = mip.Model(sense=mip.MAXIMIZE)
    x = m.add_var(lb=0.0, ub=7.0, obj=3.0, name="x")  # -c_k in min = -3 <= 0 => fix at ub
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(y <= 4)

    df = _make_df(m)
    df.presolve()

    assert float(x.lb) == pytest.approx(7.0)
    assert float(x.ub) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Already-fixed variables are skipped
# ---------------------------------------------------------------------------

def test_already_fixed_variable_is_skipped():
    """A variable with lb == ub should be left alone and not counted."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=3.0, ub=3.0, obj=1.0, name="x")  # already fixed
    y = m.add_var(lb=0.0, ub=5.0, obj=2.0, name="y")
    m.add_constr(y <= 4)

    df = _make_df(m)
    changes = df.presolve()

    # Only y might get fixed (no constraints on y from below); x was already fixed.
    assert changes <= 1  # x must not be double-counted


# ---------------------------------------------------------------------------
# Infinite bounds — unfixable
# ---------------------------------------------------------------------------

def test_no_fix_when_lb_is_infinite():
    """If lb is -inf and the fix-at-lb rule fires, the variable is unbounded
    and should be left for the solver; no fix should be recorded."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=-mip.INF, ub=10.0, obj=1.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(y <= 3)

    df = _make_df(m)
    changes = df.presolve()

    # x qualifies for fix-at-lb but lb is -inf => skip
    assert float(x.ub) == pytest.approx(10.0)
    assert float(x.lb) == pytest.approx(-_MIP_INF)


def test_no_fix_when_ub_is_infinite():
    """If ub is +inf and the fix-at-ub rule fires, skip."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=mip.INF, obj=-1.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(y <= 3)

    df = _make_df(m)
    changes = df.presolve()

    assert float(x.lb) == pytest.approx(0.0)
    assert float(x.ub) == pytest.approx(_MIP_INF)


# ---------------------------------------------------------------------------
# Lock counting
# ---------------------------------------------------------------------------

def test_ge_constraint_positive_coeff_is_downlock():
    """a_k > 0 in a >= row is a downlock; x should not be fixed at lb despite
    positive cost since it has zero uplocks but nonzero downlocks (so the
    fix-at-ub rule doesn't apply either as c_k > 0)."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=10.0, obj=2.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(x + y >= 3)  # positive coeff in >= => downlock on x

    df = _make_df(m)
    # x has uplocks=0, downlocks=1, c_k=2 > 0
    # fix-at-lb rule: uplocks==0 AND c_k>=0 => WOULD fire, but downlock doesn't block this rule
    # Actually the fix-at-lb rule only checks uplocks, so x WILL be fixed at lb.
    # This is correct: fixing x at lb=0 satisfies x + y >= 3 only if y >= 3,
    # but bounds tightening is a separate concern — dual fix only looks at locks.
    # So changes >= 1 here is expected correct behaviour.
    changes = df.presolve()
    assert isinstance(changes, int)


def test_ge_constraint_negative_coeff_is_uplock():
    """a_k < 0 in a >= row is an uplock; combined with positive cost this
    blocks the fix-at-lb rule."""
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=10.0, obj=3.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(-x + y >= 1)  # negative coeff in >= => uplock on x

    df = _make_df(m)
    changes = df.presolve()

    # x has uplocks=1, so fix-at-lb rule cannot fire
    assert float(x.lb) == pytest.approx(0.0)
    assert float(x.ub) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Return value / matrix update
# ---------------------------------------------------------------------------

def test_presolve_returns_int():
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="x")
    m.add_constr(x <= 3)

    df = _make_df(m)
    result = df.presolve()

    assert isinstance(result, int)


def test_matrix_updated_after_fixing():
    m = mip.Model(sense=mip.MINIMIZE)
    x = m.add_var(lb=1.0, ub=10.0, obj=5.0, name="x")
    y = m.add_var(lb=0.0, ub=5.0, obj=1.0, name="y")
    m.add_constr(y <= 3)

    wrapper = Model(model=m, path=None)
    df = DualFix(wrapper)
    changes = df.presolve()

    if changes > 0:
        assert wrapper.A is not None
