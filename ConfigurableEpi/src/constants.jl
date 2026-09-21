# Useful constants

# Compartmental constructors selectable by `model_type` (all S,E,I-bearing).
const COMPARTMENTAL_MODELS = Dict(:SEIRS => SEIRS, :SEIR => SEIR, :SEIS => SEIS, :SEI => SEI)

"""
    DAILY_NSSP_SUBMODELS

Submodels that run on the daily NSSP count path: one independent, from-scratch fit per Wednesday
report vintage (`origin_mode = "independent"`), dispatched to the daily runner by the entrypoint.
"""
const DAILY_NSSP_SUBMODELS = ("basic_seir_daily_nssp", "basic_seir_daily_nssp_trend")

# Defined here, ahead of the includes, because `config_schema.jl` uses it as an @option
# default and is included before `learned_hyperparams.jl`.
"""
    DEFAULT_JITTER_FLOOR_FRACTION

Minimum Liu-West jitter variance per parameter, as a fraction of that parameter's own **prior**
variance in unconstrained space.

Liu-West's jitter is proportional to the cloud's current variance, so without a floor a
degenerate cloud can never re-expand — collapse is an absorbing state (see
`_build_hyperparam_updater` in learned_hyperparams.jl for the measured evidence). This sets the
smallest step the
kernel will ever take. At 1e-3 the floor standard deviation is ~3% of the prior's, so a frozen
cloud diffuses back out over tens of steps while a healthy cloud is untouched (the floor is
applied with `max`, not added).

Too small and collapse stays effectively absorbing; too large and θ random-walks instead of
converging. Tune via `LiuWest(priors; jitter_floor_fraction = …)`.
"""
const DEFAULT_JITTER_FLOOR_FRACTION = 1.0e-3
