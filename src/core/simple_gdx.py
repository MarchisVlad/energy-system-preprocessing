"""Reconstruct the SIMPLE standard-LP model from an execute_unload GDX.

The GDX carries only data (sets + parameters + variable bounds).
Equation algebra is re-defined here in GAMSPy Python so that
model.generateFile() can write an MPS using gamspy_base's bundled
engine — bypassing the row/column limit of a local academic GAMS license.

Usage
-----
    from src.generation.simple_gdx import simple_model_from_gdx
    import gamspy as gp

    gp_model = simple_model_from_gdx("simple_4r_8760t.gdx")
    gp_model.generateFile("output/", gp.FileFormat.FixedMPS)
"""

from pathlib import Path
import gamspy as gp


def simple_model_from_gdx(gdx_path: str | Path) -> gp.Model:
    """Load a SIMPLE execute_unload GDX and reconstruct the standard-LP model.

    Parameters
    ----------
    gdx_path : str or Path
        Path to the GDX produced by ``gams simple.gms --DUMPGDX=1``.

    Returns
    -------
    gp.Model
        A GAMSPy model ready for ``generateFile()``.
    """
    c = gp.Container(load_from=str(gdx_path))

    # ------------------------------------------------------------------ sets
    rr = c["rr"]
    p = c["p"]
    s = c["s"]
    tt = c["tt"]
    e = c["e"]
    type_ = c["type"]
    t = c["t"]  # active time steps (subset of tt)
    r = c["r"]  # active regions   (subset of rr)
    rp = c["rp"]  # region-plant mapping
    rs = c["rs"]  # region-storage mapping
    net = c["net"]  # transmission links (rr1,rr2)
    ptype = c["ptype"]  # plant type mapping (rr,p,type)

    rr1 = gp.Alias(c, "rr1", rr)
    rr2 = gp.Alias(c, "rr2", rr)

    # --------------------------------------------------------------- parameters
    plant_cap = c["plant_cap"]
    demand = c["demand"]
    cost_power_generation = c["cost_power_generation"]
    cost_unserved_demand = c["cost_unserved_demand"]
    avail = c["avail"]
    total_plant_cap = c["total_plant_cap"]
    plant_emission = c["plant_emission"]
    total_emission_cap = c["total_emission_cap"]
    cost_emission = c["cost_emission"]
    storage_cap = c["storage_cap"]
    storage_efficiency = c["storage_efficiency"]
    storage_efficiency_in = c["storage_efficiency_in"]
    storage_efficiency_out = c["storage_efficiency_out"]
    storage_max_in = c["storage_max_in"]
    storage_max_out = c["storage_max_out"]
    link_cap = c["link_cap"]
    link_efficiency = c["link_efficiency"]
    cost_link_add = c["cost_link_add"]
    cost_plant_add = c["cost_plant_add"]
    cost_storage_add = c["cost_storage_add"]
    plant_max_add_cap = c["plant_max_add_cap"]
    storage_max_add_cap = c["storage_max_add_cap"]
    link_max_add_cap = c["link_max_add_cap"]

    # --------------------------------------------------------------- variables
    POWER = gp.Variable(c, "POWER", domain=[tt, rr, p], type="positive")
    FLOW = gp.Variable(c, "FLOW", domain=[tt, rr1, rr2], type="positive")
    SLACK = gp.Variable(c, "SLACK", domain=[tt, rr], type="positive")
    STORAGE_LEVEL = gp.Variable(c, "STORAGE_LEVEL", domain=[tt, rr, s], type="positive")
    STORAGE_INFLOW = gp.Variable(
        c, "STORAGE_INFLOW", domain=[tt, rr, s], type="positive"
    )
    STORAGE_OUTFLOW = gp.Variable(
        c, "STORAGE_OUTFLOW", domain=[tt, rr, s], type="positive"
    )
    PLANT_ADD_CAP = gp.Variable(c, "PLANT_ADD_CAP", domain=[rr, p], type="positive")
    STORAGE_ADD_CAP = gp.Variable(c, "STORAGE_ADD_CAP", domain=[rr, s], type="positive")
    LINK_ADD_CAP = gp.Variable(c, "LINK_ADD_CAP", domain=[rr1, rr2], type="positive")
    EMISSION_SPLIT = gp.Variable(c, "EMISSION_SPLIT", domain=[rr, e], type="positive")
    EMISSION_COST = gp.Variable(c, "EMISSION_COST", domain=[rr, e])
    ROBJ = gp.Variable(c, "ROBJ", domain=[rr])
    OBJ = gp.Variable(c, "OBJ")

    # ------------------------------------------------------------------ bounds
    PLANT_ADD_CAP.up[rr, p] = plant_max_add_cap[rr, p]
    STORAGE_ADD_CAP.up[rr, s] = storage_max_add_cap[rr, s]
    LINK_ADD_CAP.up[rr1, rr2] = link_max_add_cap[rr1, rr2]
    STORAGE_INFLOW.up[t, rr, s] = storage_max_in[rr, s]
    STORAGE_OUTFLOW.up[t, rr, s] = storage_max_out[rr, s]

    # --------------------------------------------------------------- equations
    # Domains use the active subsets (t, r, rp, rs, net) so that GAMSPy
    # generates a non-empty constraint matrix.
    eq_obj = gp.Equation(c, "eq_obj")
    eq_robj = gp.Equation(c, "eq_robj", domain=[r])
    eq_power_balance = gp.Equation(c, "eq_power_balance", domain=[t, r])
    eq_plant_capacity = gp.Equation(c, "eq_plant_capacity", domain=[t, rp])
    eq_total_plant_capacity = gp.Equation(c, "eq_total_plant_capacity", domain=[rp])
    eq_storage_balance = gp.Equation(c, "eq_storage_balance", domain=[t, rs])
    eq_storage_capacity = gp.Equation(c, "eq_storage_capacity", domain=[t, rs])
    eq_emission_region = gp.Equation(c, "eq_emission_region", domain=[r, e])
    eq_emission_cost = gp.Equation(c, "eq_emission_cost", domain=[r, e])
    eq_emission_cap = gp.Equation(c, "eq_emission_cap", domain=[e])
    eq_link_capacity = gp.Equation(c, "eq_link_capacity", domain=[t, net])

    eq_obj[...] = OBJ == (
        gp.Sum(r, ROBJ[r]) + gp.Sum(net, LINK_ADD_CAP[net] * cost_link_add[net])
    )

    eq_robj[r] = ROBJ[r] == (
        gp.Sum((t, p), POWER[t, r, p] * cost_power_generation[r, p])
        + gp.Sum(t, SLACK[t, r] * cost_unserved_demand[t])
        + gp.Sum(p, PLANT_ADD_CAP[r, p] * cost_plant_add[r, p])
        + gp.Sum(s, STORAGE_ADD_CAP[r, s] * cost_storage_add[r, s])
        + gp.Sum(e, EMISSION_COST[r, e])
    )

    eq_power_balance[t, r] = (
        gp.Sum(p, POWER[t, r, p])
        + gp.Sum((rr1, rr2), FLOW[t, rr2, r] * link_efficiency[t, rr2, r])
        - gp.Sum((rr1, rr2), FLOW[t, r, rr2])
        + gp.Sum(s, STORAGE_OUTFLOW[t, r, s] - STORAGE_INFLOW[t, r, s])
        + SLACK[t, r]
        >= demand[t, r]
    )

    eq_plant_capacity[t, rp] = (
        POWER[t, rp] <= (plant_cap[t, rp] + PLANT_ADD_CAP[rp]) * avail[t, rp]
    )

    eq_total_plant_capacity[rp] = gp.Sum(t, POWER[t, rp]) <= total_plant_cap[rp]

    # storage balance: STORAGE_LEVEL[t] = STORAGE_LEVEL[t-1] * eff + inflow*eff_in - outflow/eff_out
    # GAMSPy lag: tt.lag(1) or tt - 1 depending on version; adjust if needed
    eq_storage_balance[t, rs] = (
        STORAGE_LEVEL[t, rs]
        == STORAGE_LEVEL[t.lag(1, "circular"), rs] * storage_efficiency[rs]
        + STORAGE_INFLOW[t, rs] * storage_efficiency_in[rs]
        - STORAGE_OUTFLOW[t, rs] / storage_efficiency_out[rs]
    )

    eq_storage_capacity[t, rs] = (
        STORAGE_LEVEL[t, rs] <= storage_cap[rs] + STORAGE_ADD_CAP[rs]
    )

    eq_emission_region[r, e] = (
        gp.Sum((p, t), POWER[t, r, p] * plant_emission[r, p, e])
        <= total_emission_cap[e] * EMISSION_SPLIT[r, e]
    )

    eq_emission_cost[r, e] = (
        gp.Sum((p, t), POWER[t, r, p] * plant_emission[r, p, e]) * cost_emission[e]
        == EMISSION_COST[r, e]
    )

    eq_emission_cap[e] = gp.Sum(rr, EMISSION_SPLIT[rr, e]) <= 1

    eq_link_capacity[t, net] = FLOW[t, net] <= link_cap[t, net] + LINK_ADD_CAP[net]

    return gp.Model(
        c,
        "simple",
        equations=[
            eq_obj,
            eq_robj,
            eq_power_balance,
            eq_plant_capacity,
            eq_total_plant_capacity,
            eq_storage_balance,
            eq_storage_capacity,
            eq_emission_region,
            eq_emission_cost,
            eq_emission_cap,
            eq_link_capacity,
        ],
        problem="LP",
        sense=gp.Sense.MIN,
        objective=OBJ,
    )
