# Transmission seasonality forcing — the single implementation shared by every submodel.
#
# A submodel multiplies its force of infection by `forcing(hyper, t)`, where `t` is the model
# clock in days since the fitting window's first observation. Three modes:
#
#   cosine           annual harmonic retained for explicit legacy configurations
#   indoor_activity  the default: an empirical, location-specific curve from observed indoor activity.
#                    This package ships NO data: the caller supplies the climatology (52 knots per
#                    location) through the `climatology` keyword of the builders below, typically
#                    read with `load_indoor_activity_climatology(path)`.
#   none             a flat 1.0 control
#
# Each approach is a distinct immutable callable struct rather than one struct branching on a
# mode flag. The forcing is evaluated inside the ODE right-hand side — every RK4 substep, for
# every particle — so the approach must be resolved at build time, not per call. Submodels do this
# with a `where {F}` function barrier (see `basic_seir.jl::_build_transmission_vf`): the union
# returned by `build_seasonal_forcing` is confined to build time and the vectorfield closure
# specialises on one concrete forcing type.
#
# For the same reason, never annotate a field or argument `::SeasonalForcing` — an abstract
# field makes the ODE right-hand side dynamically dispatched.

"""
    SeasonalForcing

Supertype of the transmission-seasonality forcings. For dispatch and documentation only:
concrete subtypes are callable as `forcing(hyper, t) -> multiplier`, where `t` is days since
the start of the fitting window and `hyper` is the hyperparameter NamedTuple.
"""
abstract type SeasonalForcing end

"""Number of knots in an `indoor_activity` climatology curve — one per week of a 52-week year."""
const N_SEASON_KNOTS = 52 # weekly

const SEASONALITY_MODES = ("cosine", "indoor_activity", "none")
const SEASONALITY_FALLBACKS = ("us", "none", "error")
const INDOOR_ACTIVITY_ARTIFACT = "indoor_activity_climatology.csv2"

# ---------------------------------------------------------------------------
# The three forcings
# ---------------------------------------------------------------------------

"""
    UnitForcing()

No seasonality: the multiplier is exactly `1.0`. The control arm for judging whether a
seasonal term earns its place.
"""
struct UnitForcing <: SeasonalForcing end

@inline (::UnitForcing)(hyper, t) = 1.0

"""
    CosineForcing(day0)

The legacy learned annual harmonic,

```math
1 + a_{\\rm seas}\\cos\\{2\\pi(t + d_0 - \\varphi_{\\rm seas})/365.25\\},
```

reading `seasonal_amp` and `seasonal_phase` off `hyper` (either may be learned). `day0` is the
day-of-year of model time `t = 0`, which anchors the peak to a real calendar day rather than to
wherever the data happens to start.
"""
struct CosineForcing <: SeasonalForcing
    day0::Float64
end

@inline function (f::CosineForcing)(hyper, t)
    return 1.0 + hyper.seasonal_amp *
        cospi(2 * (t + f.day0 - hyper.seasonal_phase) / 365.25)
end

"""
    IndoorActivityForcing(curve, u0)

Empirical seasonality: a periodic curve `σ` interpolating the climatology knots, applied with
strength `κ` (read from `hyper.seasonal_kappa`) as

```math
f_{\\rm seas}(t) = 1 + \\kappa\\{\\sigma(t) - 1\\}.
```

`κ = 1` is the source paper's illustrative one-to-one forcing and `κ = 0` removes seasonality
entirely. `σ` is normalised to annual mean 1, so `f` has annual mean 1 for *any* `κ` — the
property that keeps this a reshaping of transmission rather than a level shift.

`curve` is a `DataInterpolations` interpolant over the unit year (see `build_periodic_curve`);
`u0` is the fractional position within the year of model time `t = 0`.

Only `κ` is read from `hyper`, so it is the sole point at which a `ForwardDiff.Dual` can enter;
the curve evaluation stays `Float64`.
"""
struct IndoorActivityForcing{C} <: SeasonalForcing
    curve::C
    u0::Float64
end

@inline function (f::IndoorActivityForcing)(hyper, t)
    sigma = f.curve(mod(muladd(t, 1 / 365.25, f.u0), 1.0))
    return 1.0 + hyper.seasonal_kappa * (sigma - 1.0)
end

"""
    build_periodic_curve(knots) -> CubicSpline

Fit a smooth, periodic `DataInterpolations.CubicSpline` through `knots`, knot `k` sitting at
position `u = (k - 0.5)/N` within the year, normalised to unit mean over `[0, 1)`.

Periodicity is obtained by *tiling* the knots over five consecutive years and fitting across all
of them, then only ever evaluating in the central year. `ExtrapolationType.Periodic` is not a
substitute: it repeats with period `last(t) - first(t)`, which is `(N-1)/N` of a year, not a
year, and it gives the spline no knowledge of the wrap when solving for the end segments — so
the curve is discontinuous at the year boundary. With tiling, both are exact to rounding
(≲2e-15 over a year, and a bit-exact seam).

Unit annual mean is enforced by dividing the curve by its own analytic integral over `[0, 1)`.
That is the mechanism, and it is scheme-independent: it would still hold if the interpolant were
swapped for something else.

For *this* construction the divisor is already 1, so the division is a no-op. A periodic cubic
spline through equally spaced knots integrates to exactly the knot mean, because summing its
moment equations around the period telescopes to `ΣMᵢ = 0` and leaves only the trapezoidal term;
the half-step knot offset is what gives each knot two equal half-segments there. Measured on the
committed artifact, `integral(curve, 0, 1)` and `mean(knots)` agree to the last digit.

Note that the tiling is only done _once_ at model build time, not per call to the ODE right-hand side.
The part used in the ODE is the central year of the tiling, so the ODE never sees the extra years.
"""
function build_periodic_curve(knots::AbstractVector{Float64})
    n = length(knots)
    positions = [(k - 0.5) / n + period for period in -2:2 for k in 1:n]
    tiled = repeat(collect(knots), 5)
    curve = CubicSpline(tiled, positions)
    mean_over_year = DataInterpolations.integral(curve, 0.0, 1.0)
    isfinite(mean_over_year) && mean_over_year > 0 || error(
        "seasonality curve has non-positive or non-finite annual mean $(mean_over_year)"
    )
    return CubicSpline(tiled ./ mean_over_year, positions)
end

"""
    year_fraction(d::Date) -> Float64

Position of `d` within its year, in `[0, 1)`. Leap-aware, so 1 March is at the same fraction in
a leap year as in a common year.
"""
year_fraction(d::Date) = (dayofyear(d) - 1) / daysinyear(d)

# ---------------------------------------------------------------------------
# Climatology artifact
# ---------------------------------------------------------------------------

# This package ships no data. The climatology is owned by the CALLER (a run script or pipeline),
# which either reads it from its own file with `load_indoor_activity_climatology(path)` or builds
# the `Dict` directly, and hands it to the builders through their `climatology` keyword.

"""
    load_indoor_activity_climatology(path) -> Dict{String, Vector{Float64}}

Read a caller-owned indoor-activity climatology file: one 52-knot, annual-mean-1 curve per
location (lowercase USPS abbreviation, plus `us`), as a `location,knot,value` table. `path` is
required — this package ships no data.

Deliberately a plain line reader rather than `CSV.jl`, so the module carries no CSV dependency
for a three-column table.

Validates eagerly (see [`validate_indoor_activity_climatology`](@ref)) so a malformed file fails
at load time rather than mid-solve.
"""
function load_indoor_activity_climatology(path::AbstractString)
    isfile(path) || error("indoor-activity climatology not found at `$(path)`.")
    lines = readlines(path)
    length(lines) > 1 || error("indoor-activity climatology `$(path)` is empty")
    strip(lines[1]) == "location,knot,value" || error(
        "indoor-activity climatology `$(path)` has unexpected header " *
            "`$(lines[1])` (expected `location,knot,value`)"
    )

    raw = Dict{String, Vector{Float64}}()
    for (n, line) in enumerate(@view lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 3 || error(
            "indoor-activity climatology `$(path)` line $(n + 1): expected 3 fields, " *
                "got $(length(fields))"
        )
        loc = String(fields[1])
        knot = parse(Int, fields[2])
        1 <= knot <= N_SEASON_KNOTS || error(
            "indoor-activity climatology `$(path)` line $(n + 1): knot $(knot) outside " *
                "1:$(N_SEASON_KNOTS)"
        )
        curve = get!(raw, loc) do
            fill(NaN, N_SEASON_KNOTS)
        end
        curve[knot] = parse(Float64, fields[3])
    end

    return validate_indoor_activity_climatology(raw; source = "`$(path)`")
end

"""
    validate_indoor_activity_climatology(climatology; source = "supplied climatology")
        -> Dict{String, Vector{Float64}}

Check a caller-supplied climatology and return it as a `Dict{String, Vector{Float64}}`. Every
location (a lowercase key) must carry exactly `N_SEASON_KNOTS` finite, positive knots whose mean is
1 to within `1e-6`.

The loader calls this, and so does every builder that takes a `climatology` keyword, so a curve
injected straight from a pipeline (no file) gets the same checks as one read from disk.
"""
function validate_indoor_activity_climatology(
        climatology::AbstractDict; source::AbstractString = "supplied climatology",
    )
    isempty(climatology) && error("indoor-activity climatology $(source) has no locations")
    validated = Dict{String, Vector{Float64}}()
    for (key, knots) in climatology
        loc = String(key)
        loc == lowercase(loc) || error(
            "indoor-activity climatology $(source): location key `$(loc)` must be lowercase"
        )
        length(knots) == N_SEASON_KNOTS || error(
            "indoor-activity climatology $(source): location `$(loc)` has $(length(knots)) " *
                "knots, expected $(N_SEASON_KNOTS)"
        )
        curve = collect(Float64, knots)
        all(isfinite, curve) || error(
            "indoor-activity climatology $(source): location `$(loc)` has missing or " *
                "non-finite knots"
        )
        all(>(0), curve) || error(
            "indoor-activity climatology $(source): location `$(loc)` has non-positive knots"
        )
        m = sum(curve) / N_SEASON_KNOTS
        # The forcing's annual mean is exactly the knot mean, so this is the invariant that
        # keeps `indoor_activity` a pure reshaping of transmission rather than a level shift.
        abs(m - 1) <= 1.0e-6 || error(
            "indoor-activity climatology $(source): location `$(loc)` has annual mean " *
                "$(m), expected 1"
        )
        validated[loc] = curve
    end
    return validated
end

# `indoor_activity` cannot run without a curve, and this package has none of its own.
function _require_climatology(climatology)
    climatology === nothing && error(
        "seasonality mode \"indoor_activity\" needs a climatology, and ConfigurableEpi ships no " *
            "data. Pass `climatology = load_indoor_activity_climatology(path)` (or a " *
            "`Dict(location => 52 knots)`), or select mode = \"cosine\" / \"none\"."
    )
    return validate_indoor_activity_climatology(climatology)
end

# ---------------------------------------------------------------------------
# Building a forcing from config
# ---------------------------------------------------------------------------

"""
    validate_seasonality(cfg::SeasonalityConfig)

Check a `[epi.<submodel>.seasonality]` block. Called from each submodel's `parse_epi_config`, so
a typo fails at config-validation time (and hence in the cross-language drift test) rather than
at model-build time.
"""
function validate_seasonality(cfg::SeasonalityConfig)
    cfg.mode in SEASONALITY_MODES || error(
        "seasonality mode `$(cfg.mode)` is not one of $(SEASONALITY_MODES)"
    )
    cfg.fallback in SEASONALITY_FALLBACKS || error(
        "seasonality fallback `$(cfg.fallback)` is not one of $(SEASONALITY_FALLBACKS)"
    )
    # 0 <= kappa <= 1, and the upper bound is structural rather than a numerical safeguard.
    # Because `f = 1 + kappa*(sigma - 1) == (1 - kappa) + kappa*sigma`, kappa in [0, 1] makes the
    # forcing a CONVEX COMBINATION of 1 and the curve — an interpolation between two positive
    # numbers, so positivity needs no separate check. It also has a physical reading: kappa is the
    # seasonally-forced SHARE of transmission, and `1 - kappa` the aseasonal remainder (household
    # transmission is roughly constant year-round). kappa > 1 extrapolates beyond the curve and
    # would force more than all of transmission, which is unphysical.
    isfinite(cfg.kappa) && 0 <= cfg.kappa <= 1 || error(
        "seasonality kappa must be finite and in [0, 1]; got $(cfg.kappa). kappa is the " *
            "seasonally-forced SHARE of transmission (f = (1 - kappa) + kappa*sigma), so kappa > 1 " *
            "forces more than all of it and leaves the convex hull of 1 and the curve."
    )
    return cfg
end

# Resolve a location's curve, honouring the configured fallback for locations the climatology
# does not cover (the territories; no state or `us` is ever missing).
function _resolve_curve(climatology, location, fallback)
    loc = lowercase(String(location))
    haskey(climatology, loc) && return climatology[loc]
    if fallback == "us"
        @warn "no indoor-activity curve for location; falling back to the national curve" location = loc
        return climatology["us"]
    elseif fallback == "none"
        @warn "no indoor-activity curve for location; falling back to flat seasonality" location = loc
        return fill(1.0, N_SEASON_KNOTS)
    end
    return error(
        "no indoor-activity curve for location `$(loc)` and fallback = \"error\"; " *
            "covered locations are $(sort!(collect(keys(climatology))))"
    )
end

"""
    default_seasonal_learned(cfg::SeasonalityConfig) -> Vector{Symbol}

The seasonality parameters a submodel should learn by default under `cfg`: the cosine's
amplitude and peak day-of-year, and nothing under the other modes.

`κ` is deliberately *not* learned by default — it scales an externally estimated curve, so
leaving it fixed keeps `indoor_activity` a statement about that curve rather than a free
parameter. Learn it explicitly via `learn_params = [..., "seasonal_kappa"]`.

Note the resulting asymmetry: under `cosine` the data sets the seasonal amplitude, under
`indoor_activity` it does not. A mode comparison therefore has to sweep `κ` rather than read a
single score — see `examples/seasonality_model_check.jl`.
"""
default_seasonal_learned(cfg::SeasonalityConfig) =
    cfg.mode == "cosine" ? [:seasonal_amp, :seasonal_phase] : Symbol[]

function _prior_magnitude_bound(spec::PriorSpec; probability::Real)
    spec.constraint != "unconstrained" && return prior_upper(spec; probability)
    z = quantile(Normal(), probability)
    return max(abs(spec.mean - z * spec.std), abs(spec.mean + z * spec.std))
end

# Exact maximum of the periodic cubic on [0,1]. Each interval's derivative is quadratic; values at
# its ends and midpoint determine that quadratic, so evaluating its real interior roots covers all
# spline extrema without assuming the stored knots bound cubic overshoot.
function _periodic_curve_maximum(curve)
    edges = vcat(0.0, [(k - 0.5) / N_SEASON_KNOTS for k in 1:N_SEASON_KNOTS], 1.0)
    largest = maximum(curve, edges)
    for (left, right) in zip(edges[1:(end - 1)], edges[2:end])
        width = right - left
        d0 = DataInterpolations.derivative(curve, left)
        dm = DataInterpolations.derivative(curve, left + width / 2)
        d1 = DataInterpolations.derivative(curve, right)
        # q(s) = A*s^2 + B*s + C for s in [0,1].
        C = d0
        B = 4dm - 3d0 - d1
        A = 2(d0 + d1 - 2dm)
        scale = max(abs(A), abs(B), abs(C), 1.0)
        roots = if abs(A) <= 64eps(Float64) * scale
            abs(B) <= 64eps(Float64) * scale ? Float64[] : [-C / B]
        else
            discriminant = B^2 - 4A * C
            discriminant < -64eps(Float64) * scale^2 ? Float64[] :
                [
                    (-B - sqrt(max(discriminant, 0.0))) / (2A),
                    (-B + sqrt(max(discriminant, 0.0))) / (2A),
                ]
        end
        for root in roots
            0 < root < 1 || continue
            largest = max(largest, curve(left + root * width))
        end
    end
    return largest
end

"""
    seasonal_forcing_upper_bound(cfg, fixed_amp, prior_specs, learn_params;
                                 probability = 0.95, climatology = nothing) -> Float64

A mode-aware upper bound on the seasonal multiplier used by the RK4 stability guard.

For `none` the bound is 1. For `indoor_activity` it combines the largest value over every curve
in the caller-supplied `climatology` (required for that mode) with the fixed or learned `kappa`;
pass only the curves the run can actually use to bound just those. For `cosine` it is `1 + |amplitude|`, taking amplitude
from its prior when learned and from `fixed_amp` otherwise. An empty `learn_params` means the
mode's default learned set, matching each submodel's build policy.
"""
function seasonal_forcing_upper_bound(
        cfg::SeasonalityConfig, fixed_amp::Real,
        prior_specs::AbstractDict{String, PriorSpec}, learn_params = ();
        probability::Real = 0.95, climatology = nothing,
    )
    validate_seasonality(cfg)
    requested = Symbol.(learn_params)
    learns(name) = isempty(requested) ? name in default_seasonal_learned(cfg) : name in requested

    cfg.mode == "none" && return 1.0
    if cfg.mode == "cosine"
        amp_max = if learns(:seasonal_amp)
            spec = get(prior_specs, "seasonal_amp", nothing)
            spec === nothing && error(
                "cannot bound cosine seasonality: no `seasonal_amp` prior was provided"
            )
            _prior_magnitude_bound(spec; probability)
        else
            abs(float(fixed_amp))
        end
        return 1.0 + amp_max
    end

    kappa_max = if learns(:seasonal_kappa)
        assert_seasonal_learnable(cfg, requested, prior_specs)
        prior_upper(prior_specs["seasonal_kappa"]; probability)
    else
        cfg.kappa
    end
    climatology = _require_climatology(climatology)
    sigma_max = maximum(
        knots -> _periodic_curve_maximum(build_periodic_curve(knots)), values(climatology)
    )
    return 1.0 + kappa_max * (sigma_max - 1.0)
end

"""
    assert_seasonal_learnable(cfg::SeasonalityConfig, learn_params, prior_specs = nothing)

Reject an explicit `learn_params` that names a seasonality parameter the active mode ignores.
Without this the filter would carry a dimension that cannot move the likelihood — which does not
error, it just quietly wastes particles and corrupts convergence diagnostics.

When `seasonal_kappa` is explicitly learned, `prior_specs` must constrain it to the unit interval;
checking the fixed config value alone cannot keep inferred particles inside the convex-combination
domain.
"""
function assert_seasonal_learnable(cfg::SeasonalityConfig, learn_params, prior_specs = nothing)
    inert = cfg.mode == "cosine" ? (:seasonal_kappa,) :
        cfg.mode == "indoor_activity" ? (:seasonal_amp, :seasonal_phase) :
        (:seasonal_amp, :seasonal_phase, :seasonal_kappa)
    for name in learn_params
        name in inert && error(
            "cannot learn `$(name)` under seasonality mode `$(cfg.mode)`: the forcing " *
                "never reads it, so it is unidentified. Remove it from learn_params" *
                (
                cfg.mode == "indoor_activity" ?
                    ". This mode's amplitude knob is `seasonal_kappa` — set it in " *
                    "[epi.<submodel>.seasonality] as `kappa`, or learn it explicitly. " *
                    "(`seasonal_amp`/`seasonal_phase` belong to mode = \"cosine\".)" :
                    cfg.mode == "none" ? ", or select a seasonality mode." : "."
            )
        )
    end
    if :seasonal_kappa in learn_params
        prior_specs === nothing && error(
            "learning `seasonal_kappa` requires its prior spec so the [0, 1] constraint can be checked"
        )
        spec = get(prior_specs, "seasonal_kappa", nothing)
        spec === nothing && error("cannot learn `seasonal_kappa`: no prior spec was provided")
        spec.constraint == "unit_interval" || error(
            "the learned `seasonal_kappa` prior must use constraint = \"unit_interval\"; " *
                "got $(repr(spec.constraint)). kappa is a seasonally-forced share in [0, 1]."
        )
    end
    return nothing
end


"""
    build_seasonal_forcing(cfg::SeasonalityConfig, location, start_date::Date;
                           climatology = nothing) -> SeasonalForcing

Build the seasonality forcing selected by `cfg` for `location`, anchored so that model time
`t = 0` is `start_date`.

`climatology` (a `Dict(location => 52 knots)`, e.g. from `load_indoor_activity_climatology`) is
required by `mode = "indoor_activity"` and ignored by the other modes: this package ships no data.

Returns one of `UnitForcing`, `CosineForcing` or `IndoorActivityForcing` — a small union, so
callers must put a `where {F}` function barrier between this and the ODE right-hand side (see
the note at the top of this file).
"""

function build_seasonal_forcing(
        cfg::SeasonalityConfig, location, start_date::Date; climatology = nothing,
    )
    validate_seasonality(cfg)
    cfg.mode == "none" && return UnitForcing()
    cfg.mode == "cosine" && return CosineForcing(Float64(dayofyear(start_date)))
    climatology = _require_climatology(climatology)
    knots = _resolve_curve(climatology, location, cfg.fallback)
    return IndoorActivityForcing(build_periodic_curve(knots), year_fraction(start_date))
end

"""
    build_seasonal_forcings(cfg::SeasonalityConfig, locations, start_date::Date;
                            climatology = nothing) -> Vector

One forcing per location, anchored so model time `t = 0` is `start_date`. The multi-location
counterpart of [`build_seasonal_forcing`](@ref).

Not a loop over the singular version, for two reasons. The supplied `climatology` is validated
**once** rather than once per location. And the result is a `Vector` with a CONCRETE element type (every location shares the mode, so
every element is the same forcing type), which is what keeps `forcings[x](hyper, t)` statically
dispatched inside the ODE right-hand side. A `Vector{SeasonalForcing}` would type-instability the
whole RHS — see the note at the top of this file.

The return type is still a small union across modes, so the caller needs the same `where {F}`
function barrier the singular version requires.
"""
function build_seasonal_forcings(
        cfg::SeasonalityConfig, locations, start_date::Date; climatology = nothing,
    )
    validate_seasonality(cfg)
    cfg.mode == "none" && return [UnitForcing() for _ in locations]
    cfg.mode == "cosine" &&
        return [CosineForcing(Float64(dayofyear(start_date))) for _ in locations]
    climatology = _require_climatology(climatology)
    u0 = year_fraction(start_date)
    return [
        IndoorActivityForcing(
            build_periodic_curve(_resolve_curve(climatology, location, cfg.fallback)), u0
        ) for location in locations
    ]
end
