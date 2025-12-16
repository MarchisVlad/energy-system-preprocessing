# LP PREPROCESSING BENCHMARK FOR ENERGY SYSTEM OPTIMISATION

## OVERVIEW

This repository contains a benchmarking framework for the systematic evaluation of preprocessing techniques applied to large-scale linear programming (LP) problems arising in energy system optimisation. The primary objective is to quantify the impact of presolving on problem size, numerical properties, and solver performance for industrial-scale energy models.

The framework is designed to compare preprocessing and solution pipelines in a controlled and reproducible manner. In its initial implementation, the benchmark focuses on the PaPILO LP presolver and the PIPS-IPMpp interior-point method solver. The structure of the framework allows for the straightforward integration of additional presolvers, solvers, and LP problem classes.

## SCOPE AND CONTRIBUTIONS

- Standardised benchmarking workflow for LP preprocessing
- Evaluation of presolving effects on large-scale energy system models
- Quantitative comparison of solver performance with and without preprocessing
- Modular and extensible architecture suitable for research use

## PROJECT STRUCTURE

```
energy-system-preprocessing/
├── data/                   # LP problem instances
├── packages/               # Third-party and locally built dependencies
├── presolvers/             # Presolver interfaces and configurations
│   └── scripts/            # Presolver setup and execution scripts
├── solvers/                # Solver interfaces
│   └── scripts/            # Solver setup and build scripts
└── README.md
```

## PREREQUISITES

Users must ensure the following tools and libraries are available on their system:

- wget
- cmake
- mpich2
- Boost C++ Libraries (version 1.70)

Note: On some Ubuntu distributions, Boost 1.70 is known to cause build or compatibility issues. In the current setup, Boost 1.78 is used and has been verified to compile successfully. Users are encouraged to use a local Boost build rather than relying on system-wide installations.

## PIPS-IPMpp DEPENDENCIES

PIPS-IPMpp requires at least one supported third-party sparse linear algebra library. Users should consult the original PIPS-IPMpp repository for up-to-date dependency information:

https://github.com/NCKempke/PIPS-IPMpp#

In this project, PIPS-IPMpp is built using the MA27 sparse solver from the HSL library. The MA27-based configuration provides a robust and widely used symmetric indefinite factorisation suitable for large-scale interior-point methods. Users must ensure that MA27 (or an alternative supported library) is available and correctly linked during the PIPS-IPMpp build process.

## INSTALLATION AND SETUP

The solvers/scripts/ and presolvers/scripts/ directories contain helper scripts to automate the setup process. These scripts support:

- Cloning the PaPILO and PIPS-IPMpp repositories
- Building a local version of the Boost library (recommended, as removing or modifying a system-wide Boost installation is strongly discouraged)
- Configuring and compiling PIPS-IPMpp with the selected third-party linear algebra backend

Users are encouraged to review and adapt these scripts to match their local environment, compiler, and MPI configuration.

## USAGE

### 1. Problem Preparation

Place LP problem instances in the data/ directory. Supported formats depend on the solver interfaces and may include .gms or .mps files.

### 2. Presolver Configuration

Configure PaPILO options and parameters in the presolvers/ directory. Scripts for executing presolving routines are located under presolvers/scripts/.

### 3. Solver Setup and Execution

Build and configure solver binaries using the scripts provided in solvers/scripts/. These scripts handle dependency setup and solver compilation.

### 4. Benchmark Execution

Execute the benchmarking pipeline using the provided run scripts. Both original and presolved instances are solved, and performance metrics are collected.

### 5. Result Collection

Benchmark outputs, logs, and performance metrics are written to the results directory.

## EVALUATION METRICS

The benchmark collects, where applicable:

- Number of variables and constraints before and after presolving
- Presolver runtime
- Solver runtime
- Iteration counts and convergence information
- Solver status and termination conditions

## EXTENSIBILITY

The framework is designed to support future research extensions:

- Additional presolvers can be integrated under presolvers/
- Additional solvers can be added under solvers/
- New metrics and evaluation criteria can be incorporated into the benchmarking pipeline

## REPRODUCIBILITY

All benchmarks are intended to be reproducible given fixed problem instances, presolver configurations, and solver settings. Users are encouraged to document hardware, compiler versions, MPI settings, and third-party library versions when reporting results.

## CITATION

If this software is used in academic work, please cite:

[Placeholder for citation]

## LICENSE

[Specify license information here]