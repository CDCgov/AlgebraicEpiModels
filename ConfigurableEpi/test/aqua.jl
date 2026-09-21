using Aqua

Aqua.test_all(
    ConfigurableEpi;
    stale_deps = (
        ignore = [
            # Historical ignore list, minus the script-only deps (CSV, Comonicon) that left the
            # package with run_model.jl. TODO upstream: PositiveFactorizations and the Optimization
            # pair ARE used by the module, so these entries are likely vacuous.
            :Optimization, :OptimizationOptimJL, :PositiveFactorizations,
        ],
    ),
    deps_compat = false,
    persistent_tasks = false
)
