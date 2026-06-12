# esp — Energy System Preprocessing

Benchmarking framework for studying how presolve reductions interact with the block
structure of energy-system optimisation models. Developed as part of the MEng project
*Structure-preserving preprocessing for energy-systems optimisation* (Imperial College
London, 2026).

## Overview

Energy-system models (LPs and MILPs) typically exhibit temporal and spatial block
structure that specialised solvers such as
[PIPS-IPMpp](https://github.com/NCKempke/PIPS-IPMpp) (Rehfeldt et al., 2021) exploit
by distributing subproblems and handling linking information through a
Schur-complement system. Presolving reduces model size, but can also alter or destroy
this solver-relevant structure.

`esp` provides a pipeline to:

- generate structured LP/MILP instances from the
  [SIMPLE](https://gitlab.com/beam-me/simple-pips) benchmark (Fiand et al., 2021), or
  ingest MPS files;
- apply individual or full [PaPILO](https://github.com/scipopt/papilo)
  (Gleixner et al., 2023) presolve reductions in a controlled way;
- annotate models with a block partition via hypergraph partitioning
  ([pipstools](https://gitlab.com/pips-ipmpp/pipstools) / Mt-KaHyPar);
- compare annotations before and after presolve (block counts, linking information,
  whitescore, pipstools score, clustering agreement metrics);
- re-annotate reduced models using a fixed-vertex, two-stage partitioning scheme that
  incorporates removed-element information;
- solve annotated models with PIPS-IPMpp and collect wall-time and convergence metrics.

Each pipeline stage reads from and writes to disk, so stages can run as independent
(e.g. SLURM/HPC) jobs and large models do not need to be kept in memory.

```
1. Generate / Ingest → 2. Annotate → 3. Presolve → 4. Re-annotate (fixed-vertex, optional)
                                   → 5. Compare  → 6. Solve (optional)
```

MPS is the exchange format between stages; GDX is used for annotations consumed by
PIPS-IPMpp.

## Project structure

```
energy-system-preprocessing/
├── cli/                    # `esp` command-line interface (generate, presolve, detect,
│                           #  compare, solve, ui)
├── src/
│   ├── core/               # Model / Matrix / Block abstractions, MPS reader,
│   │                       #  presolving and solving enums
│   ├── detection/          # Partitioning, annotation, scoring, comparison,
│   │                       #  visualisation; external pipstools under detection/pipstools/
│   ├── generation/         # external (modified) SIMPLE-methods model generator
│   ├── presolvers/         # PaPILO backend (subprocess wrapper) and reference
│   │                       #  Python presolver implementations
│   ├── solvers/            # PIPS-IPMpp integration and split utilities
│   ├── store/              # File-based model store and matrix cache
│   ├── utils/              # Presolve/solve handlers, annotation ops, plotting
│   └── config.py           # Paths: data root, external tool binaries
├── tools/                  # PyQt6 constraint-matrix analyser GUI, PaPILO log parser,
│                           #  metadata generation
├── experiments/            # Jupyter notebooks: setup, presolve, detection, annotation,
│                           #  scoring, solving
├── models/                 # Model format helpers and external sources (MIPLIB, GAMS)
├── resources/              # Install scripts: Boost, PaPILO, PIPS-IPMpp, pipstools
├── tests/                  # pytest suites (core, detection, presolvers, reordering)
└── data/                   # Small local artefacts; the model store lives at
                            #  /data/energy-system-preprocessing (see src/config.py)
```

## Installation

Requires Python ≥ 3.12.

```bash
pip install -e .
```

This installs the `esp` entry point and the core Python dependencies (`scipy`, `numpy`,
`matplotlib`, `networkx`, `scikit-learn`, `mip`, `polars`). The following must be
installed separately:

- **gamspy / GAMS** — model generation and GDX annotation handling (requires a GAMS
  licence; CONVERT needs a valid licence for models exceeding 5000 variables);
- **metis** — system library used by the partitioning backend;
- **PyQt6** — only needed for the GUI (`esp ui`).

### External tools

Helper scripts in `resources/` automate the third-party builds:

- `install_boost.sh` — local Boost build (1.78 verified; system-wide Boost is not
  recommended);
- `install_papilo.sh` — builds PaPILO; the binary is expected at
  `src/presolvers/papilo/build/bin/papilo`;
- `install_pips.sh` — builds PIPS-IPMpp. Requires MPI and a supported sparse linear
  algebra backend; this project uses the HSL MA27 solver (licensed, not included). See
  https://github.com/NCKempke/PIPS-IPMpp for dependency details;
- `install_pipstools.sh` — partitioning/annotation tooling (KaHyPar backend).

Paths to the data root and tool binaries are configured in `src/config.py`.

## Usage

All stages are exposed through the `esp` command and operate on the file-based model
store:

```bash
# Generate a SIMPLE instance (LP), 13 regions, quarter-year horizon, resolution 8
esp generate --name r13_res8_q --regions 13 --resolution 8 --to-period 0.25

# Generate a MIP variant with integer investment variables and a 75% capacity cap
esp generate --name r25_mip --regions 25 --integers --cap-fraction 0.75

# Annotate / detect block structure with K blocks
esp detect --model data/models/r13_res8_q -k 13

# Apply presolve steps in sequence (single PaPILO reductions or full runs)
esp presolve --model data/models/r13_res8_q --method propagation:papilo papilo

# Compare structure before and after a presolve stage
esp compare --model data/models/r13_res8_q --before original --after papilo

# Solve a stage with PIPS-IPMpp
esp solve --model data/models/r13_res8_q --stage papilo --solver pips --np 2

# Launch the interactive constraint-matrix analyser
esp ui
```

`--method name:papilo` runs PaPILO with all but the named reduction disabled, which
allows attributing structural changes to a single technique (note: final/trivial fixes
are not fully suppressed). Plain `papilo` runs the configured full reduction set.
Reference Python implementations exist for coefficient strengthening, propagation,
dual fixing, and fix-continuous; they are debugging aids, not the evaluation backend.

The notebooks under `experiments/` reproduce the experiment workflows (model
generation sweeps, presolve profiling, annotation comparison, scoring, and PIPS runs).

## Changes made to the SIMPLE methods

The SIMPLE-methods generator (`src/generation/simple-methods/`, from the
BEAM-ME project, https://gitlab.com/beam-me/simple-pips) was modified so that the
generated instances are meaningful for presolve analysis. Three
changes were made:

1. **Binding objective (`simple.gms`).** The investment (capacity-expansion) decisions
   and their linking costs are now part of the objective evaluation for the monolithic
   methods: `eq_obj` includes the regional objectives (`sum(r, ROBJ(r))`) together with
   the link expansion cost term
   (`sum(net(rr1,rr2), LINK_ADD_CAP(net) * cost_link_add(net))`) outside the
   stochastic/Lagrangian special cases.

2. **`CAP_FRACTION` parameter (`simple.gms`, default `1.0`).** A new command-line
   parameter that scales the existing plant capacity
   (`plant_cap2 = %CAP_FRACTION% * ...`), bounding installable capacity to a fraction
   of its unconstrained maximum. Varying it tightens or relaxes the feasible region,
   which controls how much presolve can reduce. Exposed via
   `esp generate --cap-fraction`.

3. **MIP variant (`simple_mip.gms`, new file).** A duplicate of the LP model in which
   the continuous investment variables `PLANT_ADD_CAP`, `STORAGE_ADD_CAP`, and
   `LINK_ADD_CAP` are redeclared under `Integer variables`, converting the instance
   into a mixed-integer programme. Their upper bounds are wrapped in `round()` to
   satisfy integrality requirements. Selected via `esp generate --integers`.

All other SIMPLE generator parameters (`FROM`, `TO`, `RESOLUTION`, `NBREGIONS`,
`METHOD`) are unchanged and map directly to `esp generate` flags.

## Evaluation metrics

- model size before/after presolve (rows, columns, nonzeros) and per-technique
  reduction counts parsed from PaPILO logs;
- **whitescore** — fraction of the matrix area outside the diagonal block regions;
- **pipstools score** — whitescore penalised by linking information (linking
  variables, global linking constraints, 2-link constraints, A0 rows), reflecting the
  expected Schur-complement cost in PIPS-IPMpp;
- clustering agreement between annotations (Adjusted Rand Index, NMI, purity) over
  surviving elements — preliminary, implementations not fully tested;
- PIPS-IPMpp wall-time and convergence status.

Scores are indicators, not predictors: part of the evaluation is identifying where
good partition scores still lead to poor solver behaviour.

## Data and reproducibility

The generated models (485 LP and 45 MILP SIMPLE instances, plus MIPLIB
instances such as `10teams` and `30n20b8`) exceed 300 GB and are not included in the
repository; all instances can be regenerated with the same pipeline. Licensed
components (GAMS, HSL MA27) and compiled third-party binaries are likewise not
distributed.

## References

- A. Gleixner, L. Gottwald, A. Hoen. *PaPILO: A Parallel Presolving Library for
  Integer and Linear Optimization with Multiprecision Support*. INFORMS Journal on
  Computing 35(6):1329–1341, 2023. https://doi.org/10.1287/ijoc.2022.0171
- D. Rehfeldt, H. Hobbie, D. Schönheit, T. Koch, D. Möst, A. Gleixner. *A massively
  parallel interior-point solver for LPs with generalized arrowhead structure, and
  applications to energy system models*. European Journal of Operational Research,
  2021.
- F. E. Curtis, J. L. Linderoth, S. J. Wright. *PIPS-IPM++: Parallel Interior Point
  Solver for Large-Scale Block Structured Problems*. Mathematical Programming
  187(2):303–331, 2021.
- F. Fiand, M. Wetzel, M. Bussieck. *SIMPLE-PIPS: The SIMPLE energy system model with
  annotation and parallel block generation*, 2021. GitLab repository:
  https://gitlab.com/beam-me/simple-pips
- M. Wetzel, S. Maher. *PIPS-IPMpp tools (pipstools)*. GitLab repository:
  https://gitlab.com/pips-ipmpp/pipstools

## License

All code in this repository is licensed under the MIT License — see [LICENSE](LICENSE).

Third-party software is not distributed with this repository: PaPILO, PIPS-IPMpp,
pipstools, and SIMPLE-methods are fetched by the `resources/` install scripts and
remain under their own licenses, while GAMS and the HSL MA27 solver must be obtained
and licensed separately by the user.
