#include "papilo/core/Presolve.hpp"
#include "papilo/io/MpsParser.hpp"
#include "papilo/io/MpsWriter.hpp"
#include "papilo/presolvers/CoefficientStrengthening.hpp"
#include "papilo/presolvers/ConstraintPropagation.hpp"
#include "papilo/presolvers/DominatedCols.hpp"
#include "papilo/presolvers/DualFix.hpp"
#include "papilo/presolvers/DualInfer.hpp"
#include "papilo/presolvers/FixContinuous.hpp"
#include "papilo/presolvers/ImplIntDetection.hpp"
#include "papilo/presolvers/ParallelColDetection.hpp"
#include "papilo/presolvers/ParallelRowDetection.hpp"
#include "papilo/presolvers/Probing.hpp"
#include "papilo/presolvers/SimpleProbing.hpp"
#include "papilo/presolvers/SimpleSubstitution.hpp"
#include "papilo/presolvers/SimplifyInequalities.hpp"
#include "papilo/presolvers/SingletonCols.hpp"
#include "papilo/presolvers/SingletonStuffing.hpp"
#include "papilo/presolvers/Sparsify.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace papilo;

using PresolverPtr = std::unique_ptr<PresolveMethod<double>>;

// Maps Python PresolvingMethod enum names to PaPILO presolver constructors.
static const std::map<std::string, std::function<PresolverPtr()>> METHOD_MAP = {
    {"CoeffTightening", [] { return std::make_unique<CoefficientStrengthening<double>>(); }},
    {"Propagation",     [] { return std::make_unique<ConstraintPropagation<double>>(); }},
    {"ColSingleton",    [] { return std::make_unique<SingletonCols<double>>(); }},
    {"DualFix",         [] { return std::make_unique<DualFix<double>>(); }},
    {"FixContinuous",   [] { return std::make_unique<FixContinuous<double>>(); }},
    {"ParallelCols",    [] { return std::make_unique<ParallelColDetection<double>>(); }},
    {"ParallelRows",    [] { return std::make_unique<ParallelRowDetection<double>>(); }},
    {"SimpleProbing",   [] { return std::make_unique<SimpleProbing<double>>(); }},
    {"DoubleToNEq",     [] { return std::make_unique<SimpleSubstitution<double>>(); }},
    {"SimpifyIneq",     [] { return std::make_unique<SimplifyInequalities<double>>(); }},
    {"Stuffing",        [] { return std::make_unique<SingletonStuffing<double>>(); }},
    {"DomCol",          [] { return std::make_unique<DominatedCols<double>>(); }},
    {"DualInfer",       [] { return std::make_unique<DualInfer<double>>(); }},
    {"ImplInt",         [] { return std::make_unique<ImplIntDetection<double>>(); }},
    {"Probing",         [] { return std::make_unique<Probing<double>>(); }},
    {"Sparsify",        [] { return std::make_unique<Sparsify<double>>(); }},
};

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: papilo_handler <input.mps> <output.mps> <method>\n";
        std::cerr << "Available methods:";
        for (const auto& kv : METHOD_MAP)
            std::cerr << " " << kv.first;
        std::cerr << "\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const std::string method_name = argv[3];

    auto it = METHOD_MAP.find(method_name);
    if (it == METHOD_MAP.end())
    {
        std::cerr << "Unknown method: " << method_name << "\n";
        return 1;
    }

    auto prob_opt = MpsParser<double>::loadProblem(input_path);
    if (!prob_opt)
    {
        std::cerr << "Failed to load: " << input_path << "\n";
        return 1;
    }

    Problem<double> problem = std::move(*prob_opt);

    Presolve<double> presolve;
    presolve.addPresolveMethod(it->second());

    PresolveOptions& opts = presolve.getPresolveOptions();
    opts.threads   = 1;
    opts.maxrounds = 1;

    PresolveResult<double> result = presolve.apply(problem);

    if (result.status == PresolveStatus::kInfeasible ||
        result.status == PresolveStatus::kUnbndOrInfeas)
    {
        std::cerr << "Presolve detected infeasibility.\n";
        return 1;
    }

    MpsWriter<double>::writeProb(
        output_path, problem,
        result.postsolve.origrow_mapping,
        result.postsolve.origcol_mapping
    );

    return 0;
}
