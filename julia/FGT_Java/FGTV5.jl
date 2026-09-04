module FGTV5

using SpecialFunctions: erfc
using LinearAlgebra: dot

export FGT,
       TRAPEZOIDAL, EXPONENTIAL,
       set_first_step_sources!,
       set_convolution_step_sources!,
       step_sources_to_targets!,
       sources_to_bins!,
       targets_to_probability,
       sources_to_probability,
       target_logwealth,
       reset_financial_parameters!,
       zero_probability

const TRAPEZOIDAL = :trapezoidal
const EXPONENTIAL = :exponential
const EPS = 5e-9
const WMULT = 5
const MAX_BIN_ARRAY = 5000
const LOG_MINUS_ONE = 0.0 + pi * im

"""Signed log. Negative real wealth is represented by log(abs(w)) + i*pi."""
function slog(w::Real)
    w > 0 && return ComplexF64(log(w), 0.0)
    w < 0 && return ComplexF64(log(-w), pi)
    return ComplexF64(-Inf, 0.0)
end

"""Return +1 for the positive signed-log branch and -1 for the negative branch."""
signbranch(z::Complex) = abs(imag(z)) <= EPS ? 1 : -1

"""Scale only the real part while snapping a tiny imaginary part to zero."""
function scalereal(z::Complex, a::Real)
    y = abs(imag(z)) <= EPS ? 0.0 : imag(z)
    return ComplexF64(real(z) * a, y)
end

"""Convert signed-log wealth back to a real signed wealth value."""
wealth(z::Complex) = real(exp(z))

mutable struct Bin
    sgn::Int
    center::ComplexF64
    effcenter::ComplexF64
    probs::Vector{Float64}
    nearfield::Matrix{ComplexF64}
    logwealth::Vector{ComplexF64}
    weights::Vector{Float64}
    exp_args::Vector{Float64}
    consumption::Float64
    prob_tot::Float64
    quadrature_weight::Float64
    is_zero_target::Bool
end

function Bin()
    Bin(1, 100.0 + 0im, 100.0 + 0im, [0.0], zeros(ComplexF64, 1, 1),
        [100.0 + 0im], [1.0], Float64[], 0.0, 0.0, 1.0, false)
end

mutable struct Interaction
    target_idx::Int
    src_ctr::ComplexF64
    distance::Float64
    shift_dist::Float64
    orig_modes::Vector{ComplexF64}
end

mutable struct InteractionQueue
    interactions::Vector{Interaction}
    target_idx::Int
    trg_ctr::ComplexF64
    trg_eff::ComplexF64
    consumption::Float64
    shift_sgn::Int
    cumulative_modes::Vector{ComplexF64}
end

mutable struct FGT
    P::Int
    PNTS::Int
    R::Int
    K::Float64
    sigma::Float64
    mu::Float64
    T::Float64
    minP::Int
    minN::Int
    quadrature::Symbol

    delta::Float64
    width::Float64
    inc::Float64
    mean::Float64
    wave::Float64
    swave::Float64
    radius::Float64

    pshift::Vector{ComplexF64}
    nshift::Vector{ComplexF64}
    mode_scale::Vector{Float64}

    source_bins::Vector{Bin}
    target_bins::Vector{Bin}
    zero_source_index::Int
    zero_target_index::Int
    consumption_target_index::Int
    zprob::Float64
    log_source_wealth::Vector{ComplexF64}
end

function FGT(p::Int, pnts::Int, r::Int, k::Real,
             sigma::Real, mu::Real, T::Real,
             minP::Int, minN::Int;
             quadrature::Symbol=TRAPEZOIDAL)
    p > 0 || throw(ArgumentError("p must be positive"))
    pnts > 0 || throw(ArgumentError("pnts must be positive"))
    quadrature in (TRAPEZOIDAL, EXPONENTIAL) ||
        throw(ArgumentError("quadrature must be :trapezoidal or :exponential"))

    delta = 2.0 * sigma^2 * T
    width = sqrt(delta)
    inc = width / pnts
    mean = (mu - sigma^2 / 2.0) * T
    wave = float(k) / p
    swave = wave / width
    radius = floor(pnts / 2) * inc

    pshift = ComplexF64[]
    nshift = ComplexF64[]
    mode_scale = Float64[]
    for j in 0:p
        z = exp(im * j * wave)
        push!(pshift, z)
        push!(nshift, conj(z))
        denom = j == 0 ? 4sqrt(pi) : 2sqrt(pi)
        push!(mode_scale, wave / (exp((j * wave / 2)^2) * denom))
    end

    return FGT(p, pnts, r, float(k), float(sigma), float(mu), float(T),
               minP, minN, quadrature, delta, width, inc, mean, wave, swave,
               radius, pshift, nshift, mode_scale, Bin[], Bin[], 1, 1, -1,
               0.0, ComplexF64[])
end

function reset_financial_parameters!(f::FGT, sigma::Real, mu::Real, T::Real)
    oldinc = f.inc
    f.delta = 2.0 * sigma^2 * T
    f.width = sqrt(f.delta)
    f.inc = f.width / f.PNTS
    changed = oldinc != f.inc
    f.mean = (mu - sigma^2 / 2.0) * T
    f.sigma = float(sigma)
    f.mu = float(mu)
    f.T = float(T)
    f.swave = f.wave / f.width
    f.radius = floor(f.PNTS / 2) * f.inc
    return changed
end

"""Jacobian d log(W+c) / d log(W)."""
dlogw(consumption::Real, target::Complex) = abs(1.0 + consumption / real(exp(target)))

function scale_target_to_source(f::FGT, w::Complex, weight::Real, consumption::Real)
    base = sqrt(pi) * f.PNTS * dlogw(consumption, w)
    return f.quadrature == EXPONENTIAL ? base / weight : base
end

# -----------------------------------------------------------------------------
# Exponential quadrature helpers
# -----------------------------------------------------------------------------

wealthfun(x::Real) = 1.0 + x - exp(-x)
weightfun(x::Real) = 1.0 + exp(-x)

function inverse_wealthfun(y::Real)
    y == 0 && return 0.0
    x = if y > 0
        y / 2.0
    else
        t = log(-y)
        t <= 1 ? -0.5 : -t
    end
    for _ in 1:100
        f = wealthfun(x) - y
        fp = 1.0 + exp(-x)
        step = f / fp
        x -= step
        abs(step) <= 1e-8 && break
    end
    return x
end

get_exp_arg(z::Complex) = inverse_wealthfun(real(z))
get_exp_arg(x::Real) = inverse_wealthfun(x)

# -----------------------------------------------------------------------------
# Range and target-grid construction
# -----------------------------------------------------------------------------

function active_source_range(f::FGT)
    isempty(f.source_bins) && return nothing
    lo = findfirst(b -> b.prob_tot >= EPS, f.source_bins)
    hi = findlast(b -> b.prob_tot >= EPS, f.source_bins)
    (lo === nothing || hi === nothing) && return nothing
    return lo, hi
end

function target_range(f::FGT, consumption::Real)
    active = active_source_range(f)
    active === nothing && return nothing
    lo, hi = active
    blo = f.source_bins[lo]
    bhi = f.source_bins[hi]

    zlo = exp(blo.logwealth[1] + f.mean - blo.sgn * f.R * f.width) - consumption
    tlo = log(zlo)
    rlo = if signbranch(tlo) == 1
        max(floor(real(tlo) / f.width), f.minP)
    else
        f.minN == typemin(Int) ? f.minP : max(ceil(real(tlo) / f.width), f.minN)
    end
    lower = f.minN == typemin(Int) ? ComplexF64(rlo, 0.0) : ComplexF64(rlo, imag(tlo))

    zhi = exp(bhi.logwealth[end] + f.mean + bhi.sgn * f.R * f.width) - consumption
    thi = log(zhi)
    signbranch(thi) < signbranch(lower) && return nothing
    rhi = if signbranch(thi) == 1
        f.minP == typemax(Int) ? f.minN : max(ceil(real(thi) / f.width), f.minP)
    else
        max(floor(real(thi) / f.width), f.minN)
    end
    upper = f.minP == typemax(Int) ? ComplexF64(rhi, pi) : ComplexF64(rhi, imag(thi))
    return lower, upper
end

function target_grid_trapezoidal(f::FGT, center::ComplexF64, c::Float64)
    sgn = signbranch(center)
    vals = Vector{ComplexF64}(undef, f.PNTS)
    vals[1] = center - sgn * f.radius
    for i in 2:f.PNTS
        vals[i] = vals[i-1] + sgn * f.inc
    end
    weights = ones(Float64, length(vals))
    return vals, weights, Float64[]
end

function target_grid_exponential(f::FGT, center::ComplexF64, c::Float64,
                                 lastbin::Union{Nothing,Bin})
    sgn = signbranch(center)
    edge = center - sgn * f.width / 2
    a0 = if lastbin === nothing || lastbin.sgn != sgn || isempty(lastbin.exp_args)
        get_exp_arg(edge)
    else
        lastbin.exp_args[end] + sgn * f.inc
    end

    aedge = get_exp_arg(center + sgn * f.width / 2)
    n = max(1, round(Int, sgn * (aedge - a0) / f.inc))
    args = [a0 + sgn * f.inc * j for j in 0:n-1]
    vals = ComplexF64[]
    for a in args
        y = wealthfun(a)
        push!(vals, sgn == 1 ? ComplexF64(y, 0.0) : ComplexF64(y, pi))
    end
    weights = weightfun.(args)
    return vals, weights, args
end

function make_target_bin(f::FGT, center::ComplexF64, consumption::Real;
                         lastbin::Union{Nothing,Bin}=nothing)
    c = float(consumption)
    sgn = signbranch(center)
    vals, weights, args = f.quadrature == TRAPEZOIDAL ?
        target_grid_trapezoidal(f, center, c) :
        target_grid_exponential(f, center, c, lastbin)

    effcenter = scalereal(log(exp(center) + c), 1.0)
    near = Matrix{ComplexF64}(undef, length(vals), f.P + 1)
    for i in eachindex(vals)
        teff = log(exp(vals[i]) + c)
        if abs(real(teff)) <= EPS
            teff = signbranch(teff) == 1 ? 0.0 + 0im : LOG_MINUS_ONE
        end
        sd = real(teff - center) * f.swave
        for k in 0:f.P
            near[i, k+1] = exp(im * sd * k)
        end
    end

    iszero = any(abs(wealth(v) + c) <= max(EPS, abs(wealth(v)) * 1e-12) for v in vals)
    return Bin(sgn, center, effcenter, zeros(length(vals)), near, vals, weights,
               args, c, 0.0, sqrt(pi) * f.PNTS, iszero)
end

function create_target_bins!(f::FGT, consumption::Real)
    lim = target_range(f, consumption)
    if lim === nothing
        f.target_bins = Bin[]
        f.zero_target_index = 1
        f.consumption_target_index = -1
        return
    end
    lower, upper = lim

    mixed_sources = 1 < f.zero_source_index <= length(f.source_bins)
    minN = mixed_sources ? floor(Int, real(f.source_bins[f.zero_source_index-1].center - f.radius) / f.width) : f.minN
    minP = mixed_sources ? floor(Int, real(f.source_bins[f.zero_source_index].center - f.radius) / f.width) : f.minP

    s0 = signbranch(lower)
    edge = if s0 < 0 && real(lower) > minN
        scalereal(lower, f.width) - f.inc
    elseif s0 < 0
        ComplexF64(minP * f.width, 0.0)
    elseif real(lower) > minP
        scalereal(lower, f.width)
    else
        ComplexF64(minP * f.width, 0.0)
    end

    width_for_center = f.quadrature == TRAPEZOIDAL ? f.radius : f.width / 2
    bins = Bin[]
    f.zero_target_index = 1
    f.consumption_target_index = -1
    lastbin = nothing
    sgn = signbranch(edge)

    while true
        center = edge + sgn * width_for_center
        b = make_target_bin(f, ComplexF64(center), consumption; lastbin=lastbin)
        push!(bins, b)
        b.is_zero_target && (f.consumption_target_index = length(bins))
        lastbin = b

        # Detect the consumption singularity from bin endpoints even when no point hits it exactly.
        w1 = wealth(b.logwealth[1]) + consumption
        w2 = wealth(b.logwealth[end]) + consumption
        if w1 == 0 || w2 == 0 || signbit(w1) != signbit(w2)
            f.consumption_target_index = length(bins)
        end

        edge = f.quadrature == TRAPEZOIDAL ?
            b.center + sgn * (f.width - width_for_center) :
            b.center + sgn * width_for_center

        # Cross from the negative signed-log branch to the positive one.
        if sgn < 0 && real(edge) < minN * f.width
            edge = ComplexF64(minP * f.width, 0.0)
            sgn = 1
            f.zero_target_index = length(bins) + 1
            lastbin = nothing
        end

        # Stop once the upper range has been covered.
        usgn = signbranch(upper)
        if sgn == usgn
            endpoint = real(edge + sgn * f.width)
            limitval = usgn * real(upper) * f.width
            if usgn * (limitval - endpoint) <= 0
                break
            end
        end
        length(bins) > MAX_BIN_ARRAY && error("Too many target bins")
    end
    f.target_bins = bins
end

# -----------------------------------------------------------------------------
# Target to source conversion
# -----------------------------------------------------------------------------

function reuse_target_for_source!(f::FGT, b::Bin)
    b.prob_tot = 0.0
    for j in eachindex(b.probs)
        sc = scale_target_to_source(f, b.logwealth[j], b.weights[j], b.consumption)
        b.probs[j] = sc >= EPS ? b.probs[j] / sc : 0.0
        b.prob_tot += b.probs[j]
    end
    return b
end

function rebuild_source_bins(f::FGT, bins::Vector{Bin})
    pts = Tuple{ComplexF64,Float64,Float64,Float64}[]
    for b in bins, j in eachindex(b.logwealth)
        push!(pts, (b.logwealth[j], b.probs[j], b.weights[j], b.consumption))
    end
    isempty(pts) && return Bin[], 1

    src = Bin[]
    i = 1
    zeroidx = 1
    oldsgn = signbranch(pts[1][1])
    while i <= length(pts)
        firstw = pts[i][1]
        sgn = signbranch(firstw)
        top = firstw + sgn * f.width
        vals = ComplexF64[]
        probs = Float64[]
        weights = Float64[]
        c = pts[i][4]
        while i <= length(pts)
            w, p, wt, cc = pts[i]
            signbranch(w) == sgn || break
            sgn * real(w - top) <= EPS || break
            sc = scale_target_to_source(f, w, wt, cc)
            push!(vals, w)
            push!(probs, sc >= EPS ? p / sc : 0.0)
            push!(weights, wt)
            i += 1
        end
        center = firstw + sgn * (f.quadrature == TRAPEZOIDAL ? f.radius : f.width / 2)
        b = Bin(sgn, center, center, probs, zeros(ComplexF64, 0, 0), vals, weights,
                Float64[], c, sum(probs), 1.0, false)
        push!(src, b)
        if sgn != oldsgn
            zeroidx = length(src)
            oldsgn = sgn
        end
    end
    return src, zeroidx
end

function sources_to_bins!(f::FGT, financial_params_changed::Bool=false)
    isempty(f.target_bins) && return f
    lo = findfirst(b -> target_bin_probability(f, b) >= EPS, f.target_bins)
    hi = findlast(b -> target_bin_probability(f, b) >= EPS, f.target_bins)
    (lo === nothing || hi === nothing) && return f
    active = f.target_bins[lo:hi]

    if !financial_params_changed
        f.source_bins = [reuse_target_for_source!(f, b) for b in active]
        f.zero_source_index = clamp(f.zero_target_index - lo + 1, 1, length(f.source_bins) + 1)
    else
        f.source_bins, f.zero_source_index = rebuild_source_bins(f, active)
    end
    f.log_source_wealth = reduce(vcat, (b.logwealth for b in f.source_bins); init=ComplexF64[])
    return f
end

# -----------------------------------------------------------------------------
# Initial condition
# -----------------------------------------------------------------------------

function initial_target_points(f::FGT, initial_wealth::Real, consumption::Real)
    source = slog(initial_wealth) + f.mean
    lo = source - f.R * f.width
    hi = source + f.R * f.width

    # Map the pre-consumption transition interval back to post-consumption wealth.
    wlo = real(exp(lo)) - consumption
    whi = real(exp(hi)) - consumption
    a, b = minmax(wlo, whi)

    vals = ComplexF64[]
    probs = Float64[]
    weights = Float64[]
    if a < 0
        xmin = max(log(abs(min(a, -EPS))), f.minN * f.width)
        xmax = log(abs(min(b, -EPS)))
        xs = collect(xmin:-f.inc:xmax)
        for x in xs
            t = ComplexF64(x, pi)
            te = log(exp(t) + consumption)
            push!(vals, t)
            push!(probs, exp(-(real((te - source) / f.width))^2))
            push!(weights, 1.0)
        end
    end
    if b > 0
        xmin = log(max(a, EPS))
        xmax = max(log(b), f.minP * f.width)
        xs = collect(max(xmin, f.minP * f.width):f.inc:xmax)
        for x in xs
            t = ComplexF64(x, 0.0)
            te = log(exp(t) + consumption)
            push!(vals, t)
            push!(probs, exp(-(real((te - source) / f.width))^2))
            push!(weights, 1.0)
        end
    end
    order = sortperm(vals; by=z -> (signbranch(z), signbranch(z) * real(z)))
    return vals[order], probs[order], weights[order]
end

function set_first_step_sources!(f::FGT, initial_wealth::Real, consumption::Real,
                                 sigma::Real=f.sigma, mu::Real=f.mu)
    reset_financial_parameters!(f, sigma, mu, 1.0)
    vals, dens, weights = initial_target_points(f, initial_wealth, consumption)
    isempty(vals) && (f.source_bins = Bin[]; return f)

    # Package the first-step density as a temporary target, then use the same
    # target-to-source conversion as subsequent periods.
    temp = Bin(signbranch(vals[1]), vals[1], vals[1], dens,
               zeros(ComplexF64, 0, 0), vals, weights, Float64[], float(consumption),
               0.0, max(1.0, length(vals) / f.PNTS), false)
    f.target_bins = [temp]
    f.source_bins, f.zero_source_index = rebuild_source_bins(f, f.target_bins)
    f.log_source_wealth = copy(vals)
    return f
end

function set_convolution_step_sources!(f::FGT, convolution::AbstractMatrix,
                                       consumption::Real,
                                       sigma::Real=f.sigma, mu::Real=f.mu)
    size(convolution, 2) >= 2 || throw(ArgumentError("convolution needs columns probability, wealth"))
    reset_financial_parameters!(f, sigma, mu, 1.0)
    vals = [slog(convolution[i,2]) for i in axes(convolution,1)]
    probs = Float64.(convolution[:,1])
    weights = ones(length(vals))
    temp = Bin(signbranch(vals[1]), vals[1], vals[1], probs,
               zeros(ComplexF64, 0, 0), vals, weights, Float64[], float(consumption),
               0.0, max(1.0, length(vals) / f.PNTS), false)
    f.target_bins = [temp]
    f.source_bins, f.zero_source_index = rebuild_source_bins(f, f.target_bins)
    f.log_source_wealth = copy(vals)
    return f
end

# -----------------------------------------------------------------------------
# Fast Gaussian interactions
# -----------------------------------------------------------------------------

function interaction_distance(f::FGT, source::Complex, target::Complex)
    signbranch(source) == signbranch(target) || return Inf
    return floor((real(target - source) - f.mean) / f.width)
end

interaction_distance(f::FGT, sb::Bin, tb::Bin) = interaction_distance(f, sb.center, tb.effcenter)

function make_interaction(f::FGT, target_idx::Int, source_idx::Int)
    sb = f.source_bins[source_idx]
    tb = f.target_bins[target_idx]
    d = interaction_distance(f, sb, tb)
    modes = zeros(ComplexF64, f.P + 1)
    modes[1] = sb.prob_tot * f.mode_scale[1]
    for k in 1:f.P
        acc = 0.0 + 0im
        for j in eachindex(sb.logwealth)
            phase = (real(tb.center - sb.logwealth[j]) - f.mean) * f.swave * k
            acc += exp(im * phase) * sb.probs[j]
        end
        modes[k+1] = acc * f.mode_scale[k+1]
    end
    return Interaction(target_idx, sb.center, d, 0.0, modes)
end

function clear!(q::InteractionQueue)
    empty!(q.interactions)
    q.target_idx = -1
    fill!(q.cumulative_modes, 0.0 + 0im)
    return q
end

function queue_for(f::FGT)
    InteractionQueue(Interaction[], -1, 0im, 0im, 0.0, 0,
                     zeros(ComplexF64, f.P + 1))
end

function insert!(f::FGT, q::InteractionQueue, target_idx::Int, source_idx::Int)
    tb = f.target_bins[target_idx]
    q.shift_sgn = signbranch(tb.center)
    q.consumption = tb.consumption
    q.trg_ctr = tb.center
    q.trg_eff = tb.effcenter
    q.target_idx = target_idx
    it = make_interaction(f, target_idx, source_idx)
    q.cumulative_modes .+= it.orig_modes
    push!(q.interactions, it)
    return q
end

function shift_queue!(f::FGT, q::InteractionQueue, next_target_idx::Int)
    q.target_idx < 0 && return q
    tb = f.target_bins[next_target_idx]
    oldctr = q.trg_ctr
    cntr_dist = abs(real(oldctr - tb.center) / f.width)
    q.trg_ctr = tb.center
    q.trg_eff = tb.effcenter
    q.shift_sgn = signbranch(tb.center)

    keep = Interaction[]
    removed = zeros(ComplexF64, f.P + 1)
    for it in q.interactions
        it.distance = interaction_distance(f, it.src_ctr, q.trg_eff)
        if abs(it.distance) > f.R
            for k in 0:f.P
                phase = exp(im * q.shift_sgn * k * f.wave * it.shift_dist)
                removed[k+1] += it.orig_modes[k+1] * phase
            end
        else
            it.shift_dist += cntr_dist
            it.target_idx = next_target_idx
            push!(keep, it)
        end
    end
    q.cumulative_modes .-= removed
    for k in 0:f.P
        q.cumulative_modes[k+1] *= exp(im * q.shift_sgn * k * f.wave * cntr_dist)
    end
    q.interactions = keep
    q.target_idx = next_target_idx
    return q
end

function shift_across_zero!(f::FGT, q::InteractionQueue, next_target_idx::Int)
    q.target_idx < 0 && return q
    old = q.trg_ctr
    tb = f.target_bins[next_target_idx]
    q.shift_sgn = signbranch(tb.center)
    q.trg_ctr = tb.center
    q.trg_eff = tb.effcenter
    delta = real(tb.center - old)

    for k in 0:f.P
        q.cumulative_modes[k+1] *= exp(im * delta * k * f.swave)
    end
    for it in q.interactions
        for k in 0:f.P
            undo = exp(-im * q.shift_sgn * k * f.wave * it.shift_dist)
            move = exp(im * delta * k * f.swave)
            it.orig_modes[k+1] *= undo * move
        end
        it.shift_dist = 0.0
        it.target_idx = next_target_idx
        it.distance = interaction_distance(f, it.src_ctr, q.trg_eff)
    end
    q.target_idx = next_target_idx
    return q
end

function interactions_to_target!(f::FGT, q::InteractionQueue, target_idx::Int)
    tb = f.target_bins[target_idx]
    fill!(tb.probs, 0.0)
    for i in eachindex(tb.probs)
        z = 0.0 + 0im
        for k in 0:f.P
            z += q.cumulative_modes[k+1] * tb.nearfield[i, k+1]
        end
        tb.probs[i] = max(0.0, 2real(z))
        tb.probs[i] < EPS && (tb.probs[i] = 0.0)
    end
    return tb
end

function direct_convolution!(f::FGT, target_idx::Int)
    tb = f.target_bins[target_idx]
    fill!(tb.probs, 0.0)
    for i in eachindex(tb.logwealth)
        teff = scalereal(log(exp(tb.logwealth[i]) + tb.consumption), 1.0)
        p = 0.0
        for sb in f.source_bins
            signbranch(sb.center) == signbranch(teff) || continue
            for j in eachindex(sb.logwealth)
                d = (real(teff - sb.logwealth[j]) - f.mean) / f.width
                abs(d) <= f.R && (p += sb.probs[j] * exp(-d^2))
            end
        end
        tb.probs[i] = p < EPS ? 0.0 : p
    end
    return tb
end

function step_sources_to_targets!(f::FGT, consumption::Real)
    create_target_bins!(f, consumption)
    isempty(f.target_bins) && return f

    q = queue_for(f)
    srcnext = 1
    prevsgn = signbranch(f.target_bins[1].center)

    for ti in eachindex(f.target_bins)
        tb = f.target_bins[ti]
        if ti > 1
            if signbranch(tb.center) != prevsgn
                shift_across_zero!(f, q, ti)
            else
                shift_queue!(f, q, ti)
            end
            prevsgn = signbranch(tb.center)
        end

        # Advance past sources that lie wholly to the left of the active window.
        while srcnext <= length(f.source_bins)
            d = interaction_distance(f, f.source_bins[srcnext], tb)
            if isinf(d)
                # Source and effective target are on different signed branches.
                if signbranch(f.source_bins[srcnext].center) < signbranch(tb.effcenter)
                    srcnext += 1
                    continue
                end
                break
            elseif d < -f.R
                srcnext += 1
            else
                break
            end
        end

        # Insert all source bins that have entered the active window.
        s = srcnext
        while s <= length(f.source_bins)
            d = interaction_distance(f, f.source_bins[s], tb)
            (!isfinite(d) || abs(d) > f.R) && break
            insert!(f, q, ti, s)
            s += 1
        end
        srcnext = max(srcnext, s)

        singular = f.consumption_target_index > 0 &&
                   (ti == f.consumption_target_index || ti == f.consumption_target_index + 1)
        if singular
            direct_convolution!(f, ti)
        elseif !isempty(q.interactions)
            interactions_to_target!(f, q, ti)
        else
            fill!(tb.probs, 0.0)
        end
    end

    compute_zero_probability!(f, consumption)
    return f
end

# -----------------------------------------------------------------------------
# Zero-region probability
# -----------------------------------------------------------------------------

function compute_zero_probability!(f::FGT, consumption::Real)
    f.zprob = 0.0
    consumption == 0 && return 0.0
    isempty(f.target_bins) && return 0.0

    # The exact Java implementation has several branch-specific cases. This
    # equivalent calculation integrates each source Gaussian over the target
    # wealth interval whose post-consumption value crosses zero.
    c = float(consumption)
    cutoff = abs(c)
    cutoff <= 0 && return 0.0
    xcut = log(cutoff)

    for sb in f.source_bins
        for j in eachindex(sb.logwealth)
            src = real(sb.logwealth[j]) + f.mean
            if c > 0 && sb.sgn == 1
                # Pre-consumption positive wealth in (0,c) becomes non-positive.
                f.zprob += 0.5 * sb.probs[j] * erfc((src - xcut) / f.width)
            elseif c < 0 && sb.sgn == -1
                # Symmetric case for saving and negative wealth.
                f.zprob += 0.5 * sb.probs[j] * erfc((src - xcut) / f.width)
            end
        end
    end
    return f.zprob
end

zero_probability(f::FGT) = f.zprob

# -----------------------------------------------------------------------------
# Output utilities
# -----------------------------------------------------------------------------

function target_bin_probability(f::FGT, b::Bin)
    if f.quadrature == EXPONENTIAL
        return dot(b.probs, b.weights) / b.quadrature_weight
    end
    return sum(b.probs) / b.quadrature_weight
end

function targets_to_probability(f::FGT)
    out = Float64[]
    for b in f.target_bins
        for j in eachindex(b.probs)
            sc = scale_target_to_source(f, b.logwealth[j], b.weights[j], b.consumption)
            p = sc >= EPS ? b.probs[j] / sc : 0.0
            push!(out, p >= EPS ? p : 0.0)
        end
    end
    return out
end

sources_to_probability(f::FGT) = reduce(vcat, (b.probs for b in f.source_bins); init=Float64[])
target_logwealth(f::FGT) = reduce(vcat, (b.logwealth for b in f.target_bins); init=ComplexF64[])

end # module
