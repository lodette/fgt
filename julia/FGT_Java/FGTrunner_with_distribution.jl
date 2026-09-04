# FGTrunner.jl
# Julia port of FGTrunner.java, designed to work with FGTV5.jl.
#
# Usage
#   include("FGTV5.jl")
#   include("FGTrunner.jl")
#   using .FGTrunner
#
# The file will include FGTV5.jl automatically if FGTV5 is not already loaded
# and FGTV5.jl is in the same directory.

if !isdefined(Main, :FGTV5)
    include(joinpath(@__DIR__, "FGTV5.jl"))
end

module FGTrunner

import ..FGTV5
using Base.Threads

export Runner, hurdle, distribution, density, set_rounding!

const DT = 1.0
const K = 7.5
const PNTS = 5
const P = 12
const R = 5
const BIN_MIN = -30
const BIN_NEG = typemin(Int)
const W_EPS = 0.005

mutable struct Runner
    round_probabilities::Bool
end

Runner(; round_probabilities::Bool=true) = Runner(round_probabilities)

set_rounding!(r::Runner, flag::Bool) = (r.round_probabilities = flag; r)

round_probability(r::Runner, p::Real) = r.round_probabilities ? min(1.0, float(p)) : float(p)
round_probability(r::Runner, p::AbstractVector) = r.round_probabilities ? min.(1.0, Float64.(p)) : Float64.(p)

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

function count_sign_changes(consumption::AbstractVector{<:Real})
    length(consumption) < 2 && return 0
    lastsign = consumption[2] < 0 ? -1 : 1
    count = 0
    for i in 3:length(consumption)
        nextsign = consumption[i] < 0 ? -1 : 1
        if nextsign != lastsign
            count += 1
            lastsign = nextsign
        end
    end
    return count
end

function range_includes_zero(horizon::Int, W::Real, idx_zero::Int,
                             consumption::AbstractVector{<:Real})
    wsign = W < 0 ? -1 : 1

    if horizon > 1
        lastsign = consumption[idx_zero] < 0 ? -1 : 1
        for i in (idx_zero + 1):length(consumption)
            nextsign = consumption[i] < 0 ? -1 : 1
            nextsign != lastsign && return true
            lastsign = nextsign
        end

        tempW = float(W)
        for i in idx_zero:length(consumption)
            tempW -= consumption[i]
            nextsign = tempW < 0 ? -1 : 1
            nextsign != wsign && return true
        end
        return false
    end

    csign = consumption[idx_zero] < 0 ? 1 : -1
    return wsign + csign == 0
end

function is_trivial(consumption::AbstractVector{<:Real}, W::Real)
    scale = maximum(abs, consumption; init=0.0)
    scale == 0 && return W == 0
    return abs(W) / scale < W_EPS
end

function consumption_scale(consumption::AbstractVector{<:Real}, idx::Int,
                           current::Real=0.0)
    scale = float(current)
    for i in idx:length(consumption)
        x = abs(float(consumption[i]))
        if x > 0
            scale = scale == 0 ? x : min(scale, x)
        end
    end
    return scale
end

function effective_params(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                          idx::Int, idx_zero::Int)
    if idx_zero > 2
        μ = sum(@view mu[idx:idx_zero-1])
        σ = sqrt(sum(abs2, @view sigma[idx:idx_zero-1]))
        return μ, σ
    end
    return float(mu[idx]), float(sigma[idx])
end

function zero_consumption_period(consumption::AbstractVector{<:Real}, idx::Int)
    i = idx + 1
    while i <= length(consumption) && consumption[i] == 0
        i += 1
    end
    return i
end

function zero_wealth_period(consumption::AbstractVector{<:Real}, W::Real)
    i = 1
    while i <= length(consumption) && W == 0 && consumption[i] == 0
        i += 1
    end
    return i
end

function validate_series(mu, sigma, consumption)
    length(mu) >= length(consumption) || throw(DimensionMismatch("mu is shorter than consumption"))
    length(sigma) >= length(consumption) || throw(DimensionMismatch("sigma is shorter than consumption"))
    isempty(consumption) && throw(ArgumentError("consumption must not be empty"))
    return nothing
end

function scaled_consumption(consumption::AbstractVector{<:Real}, idx::Int, scale::Real)
    scale > 0 || throw(ArgumentError("consumption scale is zero; at least one nonzero consumption value is required"))
    out = zeros(Float64, length(consumption))
    out[idx:end] .= consumption[idx:end] ./ scale
    return out
end

"""Probability that signed wealth is at or above a hurdle.

FGTV5 stores signed wealth as complex log-wealth. Julia can recover the signed
wealth directly with `real(exp(z))`, so no special positive/negative log-hurdle
branch code is needed here.
"""
function probability_above(logwealth::AbstractVector{<:Complex},
                           probabilities::AbstractVector{<:Real},
                           hurdle_value::Real;
                           zero_prob::Real=0.0)
    length(logwealth) == length(probabilities) || throw(DimensionMismatch("wealth and probability vectors differ in length"))
    p = 0.0
    for (z, q) in zip(logwealth, probabilities)
        real(exp(z)) >= hurdle_value && (p += q)
    end
    # zprob is mass in the region collapsed around signed wealth zero. It is
    # above any non-positive hurdle and below a strictly positive hurdle.
    hurdle_value <= 0 && (p += zero_prob)
    return clamp(p, 0.0, 1.0)
end

source_logwealth(f::FGTV5.FGT) = f.log_source_wealth

# -----------------------------------------------------------------------------
# FGT steppers
# -----------------------------------------------------------------------------

function make_fgt(eff_sigma::Real, eff_mu::Real, log_bin_min::Int, log_bin_max::Int)
    return FGTV5.FGT(P, PNTS, R, K, eff_sigma, eff_mu, DT,
                   log_bin_min, log_bin_max; quadrature=FGTV5.EXPONENTIAL)
end

function step_to_horizon(mu::Real, sigma::Real, idx_zero::Int,
                         eff_sigma::Real, eff_mu::Real, horizon::Int,
                         log_bin_min::Int, consumption::AbstractVector{<:Real},
                         startW::Real)
    log_bin_max = count_sign_changes(consumption) > 1 ? log_bin_min : BIN_NEG
    f = make_fgt(eff_sigma, eff_mu, log_bin_min, log_bin_max)

    horizon > 0 && FGTV5.set_first_step_sources!(f, startW, consumption[idx_zero], sigma, mu)
    horizon > 1 && FGTV5.step_sources_to_targets!(f, consumption[idx_zero + 1])

    if horizon > 2
        for year in 3:(horizon - 1)
            FGTV5.sources_to_bins!(f, false)
            FGTV5.step_sources_to_targets!(f, consumption[idx_zero + year - 1])
        end
    end
    return f
end

function step_to_horizon(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                         idx_zero::Int, eff_sigma::Real, eff_mu::Real,
                         horizon::Int, log_bin_min::Int,
                         consumption::AbstractVector{<:Real}, startW::Real)
    log_bin_max = count_sign_changes(consumption) > 1 ? log_bin_min : BIN_NEG
    f = make_fgt(eff_sigma, eff_mu, log_bin_min, log_bin_max)

    horizon > 0 && FGTV5.set_first_step_sources!(f, startW, consumption[idx_zero], sigma[idx_zero], mu[idx_zero])
    horizon > 1 && FGTV5.step_sources_to_targets!(f, consumption[idx_zero + 1])

    if horizon > 2
        for year in 3:(horizon - 1)
            param_idx = idx_zero + year - 2
            changed = FGTV5.reset_financial_parameters!(f, sigma[param_idx], mu[param_idx], DT)
            FGTV5.sources_to_bins!(f, changed)
            FGTV5.step_sources_to_targets!(f, consumption[idx_zero + year - 1])
        end
    end
    return f
end

function record_source_hurdle!(out::Vector{Float64}, f::FGTV5.FGT, idx::Int,
                               hurdle_value::Real, scale::Real)
    out[idx] = probability_above(source_logwealth(f), FGTV5.sources_to_probability(f),
                                 hurdle_value / scale; zero_prob=FGTV5.zero_probability(f))
    return out
end

function record_target_hurdle!(out::Vector{Float64}, f::FGTV5.FGT, idx::Int,
                               hurdle_value::Real, scale::Real)
    out[idx] = probability_above(FGTV5.target_logwealth(f), FGTV5.targets_to_probability(f),
                                 hurdle_value / scale; zero_prob=FGTV5.zero_probability(f))
    return out
end

function step_to_horizon(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                         hurdles::AbstractVector{<:Real}, idx_zero::Int,
                         eff_sigma::Real, eff_mu::Real, horizon::Int,
                         hurdle_probs::Vector{Float64}, log_bin_min::Int,
                         scale::Real, consumption::AbstractVector{<:Real},
                         startW::Real)
    log_bin_max = range_includes_zero(horizon, startW, idx_zero, consumption) ? log_bin_min : BIN_NEG
    f = make_fgt(eff_sigma, eff_mu, log_bin_min, log_bin_max)

    if horizon > 0
        FGTV5.set_first_step_sources!(f, startW, consumption[idx_zero], sigma[idx_zero], mu[idx_zero])
        record_source_hurdle!(hurdle_probs, f, idx_zero, hurdles[idx_zero], scale)
    end

    if horizon > 1
        FGTV5.step_sources_to_targets!(f, consumption[idx_zero + 1])
        record_target_hurdle!(hurdle_probs, f, idx_zero + 1, hurdles[idx_zero + 1], scale)
    end

    if horizon > 2
        for year in 3:(horizon - 1)
            param_idx = idx_zero + year - 2
            changed = FGTV5.reset_financial_parameters!(f, sigma[param_idx], mu[param_idx], DT)
            FGTV5.sources_to_bins!(f, changed)
            target_idx = idx_zero + year - 1
            FGTV5.step_sources_to_targets!(f, consumption[target_idx])
            record_target_hurdle!(hurdle_probs, f, target_idx, hurdles[target_idx], scale)
        end
    end
    return f
end

# -----------------------------------------------------------------------------
# Public hurdle methods. Julia multiple dispatch replaces Java overloads.
# -----------------------------------------------------------------------------

function hurdle(r::Runner, mu::Real, sigma::Real,
                consumption::AbstractVector{<:Real}, W::Real,
                hurdle_value::Real)
    idx = zero_wealth_period(consumption, W)
    idx <= length(consumption) || return round_probability(r, W >= hurdle_value ? 1.0 : 0.0)
    idx_zero = zero_consumption_period(consumption, idx)
    idx_zero <= length(consumption) || return round_probability(r, W >= hurdle_value ? 1.0 : 0.0)

    nzero = idx_zero - idx
    eff_sigma = nzero > 0 ? sigma * sqrt(nzero) : sigma
    eff_mu = nzero > 0 ? mu * nzero : mu
    horizon = length(consumption) - idx_zero + 2
    log_bin_min = min(BIN_MIN, floor(Int, BIN_MIN * 0.2 / sigma))

    scale = 2 * consumption_scale(consumption, idx)
    c = scaled_consumption(consumption, idx, scale)
    startW = (W - consumption[idx]) / scale

    f = step_to_horizon(mu, sigma, idx_zero, eff_sigma, eff_mu,
                        horizon, log_bin_min, c, startW)
    p = probability_above(FGTV5.target_logwealth(f), FGTV5.targets_to_probability(f),
                          hurdle_value / scale; zero_prob=FGTV5.zero_probability(f))
    return round_probability(r, p)
end

hurdle(mu::Real, sigma::Real, consumption::AbstractVector{<:Real}, W::Real,
       hurdle_value::Real) = hurdle(Runner(), mu, sigma, consumption, W, hurdle_value)

function hurdle(r::Runner, mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                consumption::AbstractVector{<:Real}, W::Real,
                hurdle_value::Real)
    validate_series(mu, sigma, consumption)
    idx = zero_wealth_period(consumption, W)
    idx <= length(consumption) || return round_probability(r, W >= hurdle_value ? 1.0 : 0.0)
    idx_zero = zero_consumption_period(consumption, idx)
    idx_zero <= length(consumption) || return round_probability(r, W >= hurdle_value ? 1.0 : 0.0)

    eff_mu, eff_sigma = effective_params(mu, sigma, idx, idx_zero)
    horizon = length(consumption) - idx_zero + 2
    avg_sigma = sum(@view sigma[idx:length(consumption)]) / length(sigma)
    log_bin_min = min(BIN_MIN, floor(Int, BIN_MIN * 0.2 / avg_sigma))

    scale = 2 * consumption_scale(consumption, idx)
    c = scaled_consumption(consumption, idx, scale)
    startW = (W - consumption[idx]) / scale

    f = step_to_horizon(mu, sigma, idx_zero, eff_sigma, eff_mu,
                        horizon, log_bin_min, c, startW)
    p = probability_above(FGTV5.target_logwealth(f), FGTV5.targets_to_probability(f),
                          hurdle_value / scale; zero_prob=FGTV5.zero_probability(f))
    return round_probability(r, p)
end

hurdle(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
       consumption::AbstractVector{<:Real}, W::Real, hurdle_value::Real) =
    hurdle(Runner(), mu, sigma, consumption, W, hurdle_value)

function hurdle(r::Runner, mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                consumption::AbstractVector{<:Real}, W::Real,
                hurdles::AbstractVector{<:Real})
    validate_series(mu, sigma, consumption)
    length(hurdles) >= length(consumption) || throw(DimensionMismatch("hurdle is shorter than consumption"))

    trivialW = is_trivial(consumption, W)
    idx = zero_wealth_period(consumption, trivialW ? 0.0 : W)
    idx <= length(consumption) || return round_probability(r, Float64.(W .>= hurdles[1:length(consumption)]))
    idx_zero = zero_consumption_period(consumption, idx)
    idx_zero <= length(consumption) || return round_probability(r, Float64.(W .>= hurdles[1:length(consumption)]))

    eff_mu, eff_sigma = effective_params(mu, sigma, idx, idx_zero)
    horizon = length(consumption) - idx_zero + 2
    avg_sigma = sum(@view sigma[idx:length(consumption)]) / length(sigma)
    log_bin_min = min(BIN_MIN, floor(Int, BIN_MIN * 0.2 / avg_sigma))

    scale = 2 * consumption_scale(consumption, idx)
    c = scaled_consumption(consumption, idx, scale)
    startW = trivialW ? -consumption[idx] / scale : (W - consumption[idx]) / scale

    probs = zeros(Float64, length(consumption))
    if startW > hurdles[idx] / scale
        probs[idx:max(idx, idx_zero - 1)] .= 1.0
    end
    if trivialW && idx > 1
        probs[1:idx-1] .= W .> hurdles[1:idx-1]
    end

    f = step_to_horizon(mu, sigma, hurdles, idx_zero, eff_sigma, eff_mu,
                        horizon, probs, log_bin_min, scale, c, startW)

    # The Java routine computes the final horizon hurdle after the stepper.
    final_idx = idx_zero + horizon - 2
    if 1 <= final_idx <= length(consumption)
        record_target_hurdle!(probs, f, final_idx, hurdles[final_idx], scale)
    end

    return round_probability(r, probs)
end

hurdle(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
       consumption::AbstractVector{<:Real}, W::Real,
       hurdles::AbstractVector{<:Real}) =
    hurdle(Runner(), mu, sigma, consumption, W, hurdles)


# -----------------------------------------------------------------------------
# Horizon distribution and density
# -----------------------------------------------------------------------------

"""
    distribution([runner], mu, sigma, consumption, W) -> wealth, mass

Propagate the initial wealth `W` through the full horizon implied by
`consumption` and return the terminal distribution in the original wealth
units. `wealth[i]` is a terminal wealth grid point and `mass[i]` is its
associated probability mass. Any probability collapsed at exactly zero wealth
is returned explicitly as a point mass at `wealth == 0`.
"""
function _terminal_distribution(f::FGTV5.FGT, scale::Real)
    z = FGTV5.target_logwealth(f)
    q = Float64.(FGTV5.targets_to_probability(f))
    length(z) == length(q) || throw(DimensionMismatch("terminal wealth and probability vectors differ in length"))

    wealth = scale .* real.(exp.(z))
    mass = q

    zprob = float(FGTV5.zero_probability(f))
    if zprob > 0
        push!(wealth, 0.0)
        push!(mass, zprob)
    end

    # The signed log representation can yield grid points from two branches.
    # Sort in ordinary wealth space and combine numerically identical points.
    perm = sortperm(wealth)
    wealth = wealth[perm]
    mass = mass[perm]

    isempty(wealth) && return wealth, mass
    wout = Float64[wealth[1]]
    mout = Float64[mass[1]]
    for i in 2:length(wealth)
        if wealth[i] == wout[end]
            mout[end] += mass[i]
        else
            push!(wout, wealth[i])
            push!(mout, mass[i])
        end
    end
    return wout, mout
end

function distribution(::Runner, mu::Real, sigma::Real,
                      consumption::AbstractVector{<:Real}, W::Real)
    isempty(consumption) && return Float64[float(W)], Float64[1.0]

    idx = zero_wealth_period(consumption, W)
    idx <= length(consumption) || return Float64[float(W)], Float64[1.0]
    idx_zero = zero_consumption_period(consumption, idx)
    idx_zero <= length(consumption) || return Float64[float(W)], Float64[1.0]

    nzero = idx_zero - idx
    eff_sigma = nzero > 0 ? sigma * sqrt(nzero) : sigma
    eff_mu = nzero > 0 ? mu * nzero : mu
    horizon = length(consumption) - idx_zero + 2
    log_bin_min = min(BIN_MIN, floor(Int, BIN_MIN * 0.2 / sigma))

    scale = 2 * consumption_scale(consumption, idx)
    c = scaled_consumption(consumption, idx, scale)
    startW = (W - consumption[idx]) / scale

    f = step_to_horizon(mu, sigma, idx_zero, eff_sigma, eff_mu,
                        horizon, log_bin_min, c, startW)
    return _terminal_distribution(f, scale)
end

distribution(mu::Real, sigma::Real, consumption::AbstractVector{<:Real}, W::Real) =
    distribution(Runner(), mu, sigma, consumption, W)

function distribution(::Runner, mu::AbstractVector{<:Real},
                      sigma::AbstractVector{<:Real},
                      consumption::AbstractVector{<:Real}, W::Real)
    validate_series(mu, sigma, consumption)

    idx = zero_wealth_period(consumption, W)
    idx <= length(consumption) || return Float64[float(W)], Float64[1.0]
    idx_zero = zero_consumption_period(consumption, idx)
    idx_zero <= length(consumption) || return Float64[float(W)], Float64[1.0]

    eff_mu, eff_sigma = effective_params(mu, sigma, idx, idx_zero)
    horizon = length(consumption) - idx_zero + 2
    avg_sigma = sum(@view sigma[idx:length(consumption)]) / length(sigma)
    log_bin_min = min(BIN_MIN, floor(Int, BIN_MIN * 0.2 / avg_sigma))

    scale = 2 * consumption_scale(consumption, idx)
    c = scaled_consumption(consumption, idx, scale)
    startW = (W - consumption[idx]) / scale

    f = step_to_horizon(mu, sigma, idx_zero, eff_sigma, eff_mu,
                        horizon, log_bin_min, c, startW)
    return _terminal_distribution(f, scale)
end

distribution(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
             consumption::AbstractVector{<:Real}, W::Real) =
    distribution(Runner(), mu, sigma, consumption, W)

"""
    density([runner], mu, sigma, consumption, W) -> wealth, pdf

Return a wealth-space PDF reconstructed from the terminal quadrature masses.
Any discrete atom at exactly zero wealth is omitted because it cannot be
represented by an ordinary density. Use `distribution` when that point mass
matters.
"""
function _mass_to_density(wealth::AbstractVector{<:Real},
                          mass::AbstractVector{<:Real})
    length(wealth) == length(mass) || throw(DimensionMismatch("wealth and mass vectors differ in length"))

    # Keep only continuous grid points with positive mass. A zero point created
    # by zprob is a discrete atom rather than part of the continuous PDF.
    keep = [i for i in eachindex(wealth) if wealth[i] != 0 && mass[i] > 0]
    length(keep) < 2 && return Float64[], Float64[]

    w = Float64.(wealth[keep])
    m = Float64.(mass[keep])
    perm = sortperm(w)
    w = w[perm]
    m = m[perm]

    n = length(w)
    widths = similar(w)
    widths[1] = (w[2] - w[1]) / 2
    widths[end] = (w[end] - w[end-1]) / 2
    if n > 2
        @inbounds for i in 2:n-1
            widths[i] = (w[i+1] - w[i-1]) / 2
        end
    end

    # Guard against repeated or pathological grid points.
    good = widths .> 0
    return w[good], m[good] ./ widths[good]
end

function density(r::Runner, mu, sigma, consumption::AbstractVector{<:Real}, W::Real)
    wealth, mass = distribution(r, mu, sigma, consumption, W)
    return _mass_to_density(wealth, mass)
end

density(mu, sigma, consumption::AbstractVector{<:Real}, W::Real) =
    density(Runner(), mu, sigma, consumption, W)


# -----------------------------------------------------------------------------
# Two-stream and three-stream wrappers
# -----------------------------------------------------------------------------

function hurdle(r::Runner, mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                c0::AbstractVector{<:Real}, W0::Real,
                c1::AbstractVector{<:Real}, W1::Real,
                hurdles::AbstractVector{<:Real})
    if length(c0) >= length(c1)
        long = Float64.(c0)
        short = Float64.(c0[1:length(c1)])
        combined = copy(long)
        combined[1:length(c1)] .+= c1
        Wshort = W0
    else
        long = Float64.(c1)
        short = Float64.(c1[1:length(c0)])
        combined = copy(long)
        combined[1:length(c0)] .+= c0
        Wshort = W1
    end

    W = W0 + W1
    short_h = Float64.(hurdles[1:length(short)])

    # The Java code runs the short calculation in a thread. Threads.@spawn is
    # the direct Julia analogue and avoids a custom thread-wrapper class.
    short_task = Threads.@spawn hurdle(Runner(round_probabilities=false), mu, sigma,
                                       short, Wshort, short_h)
    probs = hurdle(Runner(round_probabilities=false), mu, sigma, combined, W,
                    Float64.(hurdles[1:length(combined)]))
    short_probs = fetch(short_task)
    probs[1:length(short_probs)] .= short_probs
    return round_probability(r, probs)
end

hurdle(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
       c0::AbstractVector{<:Real}, W0::Real,
       c1::AbstractVector{<:Real}, W1::Real,
       hurdles::AbstractVector{<:Real}) =
    hurdle(Runner(), mu, sigma, c0, W0, c1, W1, hurdles)

function hurdle(r::Runner, mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                c0::AbstractVector{<:Real}, W0::Real,
                c1::AbstractVector{<:Real}, W1::Real,
                c2::AbstractVector{<:Real}, W2::Real,
                hurdles::AbstractVector{<:Real})
    streams = [Float64.(c0), Float64.(c1), Float64.(c2)]
    wealths = [float(W0), float(W1), float(W2)]
    order = sortperm(length.(streams))
    dshort, dmedium, dlong = order

    nshort = length(streams[dshort])
    nmedium = length(streams[dmedium])
    nlong = length(streams[dlong])

    # Preserve the Java stream construction exactly. The naming is historical.
    W = wealths[dshort]
    Wmedium = W + wealths[dmedium]
    W += wealths[dmedium]
    Wshort = wealths[dlong]
    Wmedium += wealths[dlong]
    W += wealths[dlong]

    cshort = zeros(Float64, nshort)
    cmedium = zeros(Float64, nmedium)
    combined = zeros(Float64, nlong)

    for i in 1:nlong
        if i <= nshort
            cshort[i] = streams[dlong][i]
            cmedium[i] = streams[dlong][i] + streams[dshort][i]
            combined[i] = streams[dlong][i] + streams[dmedium][i] + streams[dshort][i]
        elseif i <= nmedium
            cmedium[i] = streams[dlong][i]
            combined[i] = streams[dlong][i] + streams[dmedium][i]
        else
            combined[i] = streams[dlong][i]
        end
    end

    short_h = Float64.(hurdles[1:nshort])
    medium_h = Float64.(hurdles[1:nmedium])
    long_h = Float64.(hurdles[1:nlong])

    short_task = Threads.@spawn hurdle(Runner(round_probabilities=false), mu, sigma,
                                       cshort, Wshort, short_h)
    medium_task = Threads.@spawn hurdle(Runner(round_probabilities=false), mu, sigma,
                                        cmedium, Wmedium, medium_h)
    probs = hurdle(Runner(round_probabilities=false), mu, sigma, combined, W, long_h)

    medium_probs = fetch(medium_task)
    short_probs = fetch(short_task)
    probs[1:length(medium_probs)] .= medium_probs
    probs[1:length(short_probs)] .= short_probs
    return round_probability(r, probs)
end

hurdle(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
       c0::AbstractVector{<:Real}, W0::Real,
       c1::AbstractVector{<:Real}, W1::Real,
       c2::AbstractVector{<:Real}, W2::Real,
       hurdles::AbstractVector{<:Real}) =
    hurdle(Runner(), mu, sigma, c0, W0, c1, W1, c2, W2, hurdles)

end # module FGTrunner
