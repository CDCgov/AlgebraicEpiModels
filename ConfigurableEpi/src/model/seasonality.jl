# Transmission seasonality: a multiplier `forcing(hyper, t)` on the force of infection, with `t`
# in days since the window start. Each mode is its own concrete callable, so a submodel should pass
# the forcing through a `where {F}` function barrier before capturing it in the vector field.

abstract type SeasonalForcing end

"Knots per `indoor_activity` climatology curve: one per week of the year."
const N_SEASON_KNOTS = 52
const SEASONALITY_MODES = ("cosine", "indoor_activity", "none")
const SEASONALITY_FALLBACKS = ("us", "none", "error")

"""
    UnitForcing()
    CosineForcing(day0)
    IndoorActivityForcing(curve, u0)

The three forcings: flat `1.0`; the annual harmonic
`1 + hyper.seasonal_amp * cos(2π (t + day0 - hyper.seasonal_phase) / 365.25)` with `day0` the
day-of-year of `t = 0`; and `1 + hyper.seasonal_kappa * (σ(t) - 1)` with `σ` a unit-mean periodic
spline through a location's climatology and `u0` the year fraction of `t = 0`.
"""
struct UnitForcing <: SeasonalForcing end
@inline (::UnitForcing)(hyper, t) = 1.0

struct CosineForcing <: SeasonalForcing
    day0::Float64
end
@inline (f::CosineForcing)(hyper, t) =
    1.0 + hyper.seasonal_amp * cospi(2 * (t + f.day0 - hyper.seasonal_phase) / 365.25)

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

Unit-mean periodic cubic spline through `knots`, knot `k` at `u = (k - 0.5) / N` within the year.
Periodicity comes from tiling five years and evaluating in the central one (the interpolant's own
periodic extrapolation repeats with period `(N-1)/N`); the annual mean is normalised by the
analytic integral.
"""
function build_periodic_curve(knots::AbstractVector{Float64})
    n = length(knots)
    positions = [(k - 0.5) / n + period for period in -2:2 for k in 1:n]
    tiled = repeat(collect(knots), 5)
    curve = CubicSpline(tiled, positions)
    mean_over_year = DataInterpolations.integral(curve, 0.0, 1.0)
    isfinite(mean_over_year) && mean_over_year > 0 ||
        error("seasonality curve has non-positive or non-finite annual mean $mean_over_year")
    return CubicSpline(tiled ./ mean_over_year, positions)
end

"Position of `d` within its year in `[0, 1)`, leap-aware."
year_fraction(d::Date) = (dayofyear(d) - 1) / daysinyear(d)

"""
    load_indoor_activity_climatology(path) -> Dict{String, Vector{Float64}}

Read a `location,knot,value` table of 52-knot, unit-mean curves (lowercase location keys, `us`
included). The package ships no data; the caller owns this file.
"""
function load_indoor_activity_climatology(path::AbstractString)
    isfile(path) || error("indoor-activity climatology not found at `$path`")
    lines = readlines(path)
    length(lines) > 1 || error("indoor-activity climatology `$path` is empty")
    strip(lines[1]) == "location,knot,value" ||
        error("indoor-activity climatology `$path` has header `$(lines[1])`, expected `location,knot,value`")
    raw = Dict{String, Vector{Float64}}()
    for (n, line) in enumerate(@view lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == 3 || error("indoor-activity climatology `$path` line $(n + 1): expected 3 fields")
        knot = parse(Int, fields[2])
        1 <= knot <= N_SEASON_KNOTS ||
            error("indoor-activity climatology `$path` line $(n + 1): knot $knot outside 1:$N_SEASON_KNOTS")
        curve = get!(() -> fill(NaN, N_SEASON_KNOTS), raw, String(fields[1]))
        curve[knot] = parse(Float64, fields[3])
    end
    return validate_indoor_activity_climatology(raw; source = "`$path`")
end

"""
    validate_indoor_activity_climatology(climatology; source = "supplied climatology")

Check that every (lowercase) location carries `N_SEASON_KNOTS` finite, positive knots with mean 1.
"""
function validate_indoor_activity_climatology(
        climatology::AbstractDict; source::AbstractString = "supplied climatology",
    )
    isempty(climatology) && error("indoor-activity climatology $source has no locations")
    validated = Dict{String, Vector{Float64}}()
    for (key, knots) in climatology
        loc = String(key)
        loc == lowercase(loc) || error("indoor-activity climatology $source: location `$loc` must be lowercase")
        length(knots) == N_SEASON_KNOTS ||
            error("indoor-activity climatology $source: `$loc` has $(length(knots)) knots, expected $N_SEASON_KNOTS")
        curve = collect(Float64, knots)
        all(isfinite, curve) && all(>(0), curve) ||
            error("indoor-activity climatology $source: `$loc` has non-finite or non-positive knots")
        abs(sum(curve) / N_SEASON_KNOTS - 1) <= 1.0e-6 ||
            error("indoor-activity climatology $source: `$loc` has annual mean $(sum(curve) / N_SEASON_KNOTS), expected 1")
        validated[loc] = curve
    end
    return validated
end

function _require_climatology(climatology)
    climatology === nothing && error(
        "seasonality mode \"indoor_activity\" needs a climatology and ConfigurableEpi ships no data: pass " *
            "`climatology = load_indoor_activity_climatology(path)` (or a Dict(location => 52 knots)), " *
            "or select mode = \"cosine\" / \"none\"",
    )
    return validate_indoor_activity_climatology(climatology)
end

"""
    validate_seasonality(cfg::SeasonalityConfig) -> cfg

Check the mode, the fallback and `kappa ∈ [0, 1]` (the seasonally forced share of transmission).
"""
function validate_seasonality(cfg::SeasonalityConfig)
    cfg.mode in SEASONALITY_MODES || error("seasonality mode `$(cfg.mode)` is not one of $SEASONALITY_MODES")
    cfg.fallback in SEASONALITY_FALLBACKS ||
        error("seasonality fallback `$(cfg.fallback)` is not one of $SEASONALITY_FALLBACKS")
    isfinite(cfg.kappa) && 0 <= cfg.kappa <= 1 || error(
        "seasonality kappa must be in [0, 1], got $(cfg.kappa): it is the seasonally forced share of " *
            "transmission, f = (1 - kappa) + kappa * sigma",
    )
    return cfg
end

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
    return error("no indoor-activity curve for `$loc` and fallback = \"error\"; covered: $(sort!(collect(keys(climatology))))")
end

"""
    default_seasonal_learned(cfg::SeasonalityConfig) -> Vector{Symbol}

Seasonality parameters learned by default: the cosine's amplitude and phase, nothing otherwise
(`kappa` scales an externally estimated curve, so it is opt-in via `learn_params`).
"""
default_seasonal_learned(cfg::SeasonalityConfig) =
    cfg.mode == "cosine" ? [:seasonal_amp, :seasonal_phase] : Symbol[]

function _prior_magnitude_bound(spec::PriorSpec; probability::Real)
    spec.constraint != "unconstrained" && return prior_upper(spec; probability)
    z = quantile(Normal(), probability)
    return max(abs(spec.mean - z * spec.std), abs(spec.mean + z * spec.std))
end

# Exact maximum of the periodic cubic on [0, 1]: each interval's derivative is a quadratic fixed by
# its values at the ends and midpoint, so its interior roots cover every spline extremum.
function _periodic_curve_maximum(curve)
    edges = vcat(0.0, [(k - 0.5) / N_SEASON_KNOTS for k in 1:N_SEASON_KNOTS], 1.0)
    largest = maximum(curve, edges)
    for (left, right) in zip(edges[1:(end - 1)], edges[2:end])
        width = right - left
        d0 = DataInterpolations.derivative(curve, left)
        dm = DataInterpolations.derivative(curve, left + width / 2)
        d1 = DataInterpolations.derivative(curve, right)
        C, B, A = d0, 4dm - 3d0 - d1, 2(d0 + d1 - 2dm)
        scale = max(abs(A), abs(B), abs(C), 1.0)
        roots = if abs(A) <= 64eps(Float64) * scale
            abs(B) <= 64eps(Float64) * scale ? Float64[] : [-C / B]
        else
            disc = B^2 - 4A * C
            disc < -64eps(Float64) * scale^2 ? Float64[] :
                [(-B - sqrt(max(disc, 0.0))) / (2A), (-B + sqrt(max(disc, 0.0))) / (2A)]
        end
        for root in roots
            0 < root < 1 && (largest = max(largest, curve(left + root * width)))
        end
    end
    return largest
end

"""
    seasonal_forcing_upper_bound(cfg, fixed_amp, prior_specs, learn_params = ();
                                 probability = 0.95, climatology = nothing) -> Float64

Mode-aware upper bound on the seasonal multiplier for the RK4 stability guard: 1 for `none`,
`1 + |amplitude|` for `cosine` (from the prior when learned), and the largest curve value in
`climatology` scaled by the fixed or learned `kappa` for `indoor_activity`. An empty
`learn_params` means the mode's default learned set.
"""
function seasonal_forcing_upper_bound(
        cfg::SeasonalityConfig, fixed_amp::Real, prior_specs::AbstractDict{String, PriorSpec},
        learn_params = (); probability::Real = 0.95, climatology = nothing,
    )
    validate_seasonality(cfg)
    requested = Symbol.(learn_params)
    learns(name) = isempty(requested) ? name in default_seasonal_learned(cfg) : name in requested
    cfg.mode == "none" && return 1.0
    if cfg.mode == "cosine"
        amp_max = if learns(:seasonal_amp)
            spec = get(prior_specs, "seasonal_amp", nothing)
            spec === nothing && error("cannot bound cosine seasonality: no `seasonal_amp` prior was provided")
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
    sigma_max = maximum(knots -> _periodic_curve_maximum(build_periodic_curve(knots)), values(climatology))
    return 1.0 + kappa_max * (sigma_max - 1.0)
end

"""
    assert_seasonal_learnable(cfg::SeasonalityConfig, learn_params, prior_specs = nothing)

Reject learning a seasonality parameter the active mode never reads (an unidentified dimension),
and require a `unit_interval` prior when `seasonal_kappa` is learned.
"""
function assert_seasonal_learnable(cfg::SeasonalityConfig, learn_params, prior_specs = nothing)
    inert = cfg.mode == "cosine" ? (:seasonal_kappa,) :
        cfg.mode == "indoor_activity" ? (:seasonal_amp, :seasonal_phase) :
        (:seasonal_amp, :seasonal_phase, :seasonal_kappa)
    for name in learn_params
        name in inert && error(
            "cannot learn `$name` under seasonality mode `$(cfg.mode)`: the forcing never reads it. " *
                "Remove it from learn_params" *
                (cfg.mode == "indoor_activity" ? "; this mode's amplitude knob is `seasonal_kappa`." : "."),
        )
    end
    if :seasonal_kappa in learn_params
        spec = prior_specs === nothing ? nothing : get(prior_specs, "seasonal_kappa", nothing)
        spec === nothing && error("learning `seasonal_kappa` requires its prior spec")
        spec.constraint == "unit_interval" ||
            error("the `seasonal_kappa` prior must use constraint = \"unit_interval\"; got $(repr(spec.constraint))")
    end
    return nothing
end

"""
    build_seasonal_forcing(cfg::SeasonalityConfig, location, start_date::Date; climatology = nothing)
    build_seasonal_forcings(cfg, locations, start_date; climatology = nothing) -> Vector

The forcing selected by `cfg`, anchored so model time `t = 0` is `start_date`. `climatology`
(`Dict(location => 52 knots)`) is required by `indoor_activity`. The result type depends on the
mode, so specialise the vector field on it through a `where {F}` function barrier.
"""
function build_seasonal_forcing(cfg::SeasonalityConfig, location, start_date::Date; climatology = nothing)
    validate_seasonality(cfg)
    cfg.mode == "none" && return UnitForcing()
    cfg.mode == "cosine" && return CosineForcing(Float64(dayofyear(start_date)))
    knots = _resolve_curve(_require_climatology(climatology), location, cfg.fallback)
    return IndoorActivityForcing(build_periodic_curve(knots), year_fraction(start_date))
end

build_seasonal_forcings(cfg::SeasonalityConfig, locations, start_date::Date; climatology = nothing) =
    [build_seasonal_forcing(cfg, location, start_date; climatology) for location in locations]
