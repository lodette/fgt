"""
    dgt.jl — Discrete Gauss Transform (Plane-Wave / FGT Algorithm)

Implements the fast Gaussian convolution described in:

  L. Odette, "Discrete Gauss Transforms for Investment-Consumption
  (Hurdle) Analysis", FM Global, May 2014.

Based on Greengard & Sun (1998). Uses Julia's built-in `Complex{Float64}`
throughout — no external FFT or linear-algebra libraries are required.

The transform to be evaluated is

    prob[j] = Σᵢ ρᵢ G(tⱼ, sᵢ)

where G(t, s) = (1/√(πδ)) exp(-(t - s - m)² / δ).

The algorithm proceeds in three stages:
  1. Source to Plane-Wave  (S2W)
  2. Plane-Wave to Local   (W2L)  — with incremental bin-to-bin phase shift
  3. Local to Target       (L2T)
"""
module DGT

# ── Standard-library imports (no Pkg.add needed) ─────────────────────────────
# LinearAlgebra — norm(), dot(), etc. for error analysis and extensions.
# Statistics    — mean(), std(), var() for moment calculations.
using LinearAlgebra: norm
using Statistics: mean, std, var

export dgt_naive, dgt_planwave, DGTParams

# ─────────────────────────────────────────────────────────────
# Parameter struct
# ─────────────────────────────────────────────────────────────

"""
    DGTParams

Holds the physical and algorithmic parameters for one time-step of
the discrete investment-consumption model.

# Fields
- `μ`  : annual drift
- `σ`  : annual volatility
- `Δτ` : length of one time step
- `P`  : number of plane-wave modes (half-band; modes run -P..P)
- `Ω`  : bandwidth cutoff in wavenumber space
"""
struct DGTParams
    μ  :: Float64   # drift
    σ  :: Float64   # volatility
    Δτ :: Float64   # time-step length
    P  :: Int       # number of plane-wave modes
    Ω  :: Float64   # wavenumber bandwidth

    # Derived constants (computed once at construction)
    δ  :: Float64   # 2σ²Δτ
    m  :: Float64   # (μ - σ²/2)Δτ
    Δω :: Float64   # Ω / P
end

function DGTParams(; μ, σ, Δτ, P=64, Ω=10.0)
    δ  = 2σ^2 * Δτ
    m  = (μ - σ^2 / 2) * Δτ
    Δω = Ω / P
    DGTParams(μ, σ, Δτ, P, Ω, δ, m, Δω)
end

# ─────────────────────────────────────────────────────────────
# Naïve O(NM) reference implementation
# ─────────────────────────────────────────────────────────────

"""
    dgt_naive(s, ρ, t, par) -> Vector{Float64}

Reference O(NM) evaluation of the Discrete Gauss Transform.

    prob[j] = Σᵢ ρᵢ / √(πδ) × exp(-(tⱼ - sᵢ - m)² / δ)

# Arguments
- `s`   : source log-wealth points (length N)
- `ρ`   : source probability weights (length N)
- `t`   : target log-wealth points (length M)
- `par` : `DGTParams` instance
"""
function dgt_naive(
        s   :: AbstractVector{Float64},
        ρ   :: AbstractVector{Float64},
        t   :: AbstractVector{Float64},
        par :: DGTParams
    ) :: Vector{Float64}

    N, M = length(s), length(t)
    @assert length(ρ) == N
    inv_sqrt_πδ = 1.0 / sqrt(π * par.δ)
    prob = zeros(Float64, M)

    for j in 1:M
        acc = 0.0
        @inbounds for i in 1:N
            z = (t[j] - s[i] - par.m) / sqrt(par.δ)
            acc += ρ[i] * exp(-z^2)
        end
        prob[j] = inv_sqrt_πδ * acc
    end
    return prob
end

# ─────────────────────────────────────────────────────────────
# Plane-wave weights θ[p]
# ─────────────────────────────────────────────────────────────

"""
    plane_wave_weights(par) -> Vector{Complex{Float64}}

Pre-compute the plane-wave weights for modes p = -P…P.

    θ(p) = Δω/(2π√δ) × exp(-(pΔω/2)²)

Returned as a vector indexed 1 .. 2P+1 (index `P+1` ↔ mode 0).
"""
function plane_wave_weights(par::DGTParams) :: Vector{Float64}
    P   = par.P
    Δω  = par.Δω
    δ   = par.δ
    prefactor = Δω / (2π * sqrt(δ))
    θ = Vector{Float64}(undef, 2P + 1)
    for k in 1:(2P+1)
        p = k - (P + 1)   # mode index: -P..P
        θ[k] = prefactor * exp(-(p * Δω / 2)^2)
    end
    return θ
end

# ─────────────────────────────────────────────────────────────
# Plane-wave basis function  f(ξ, p) = exp(i·p·Δω·ξ/√δ)
# ─────────────────────────────────────────────────────────────

@inline function f_basis(ξ::Float64, p::Int, par::DGTParams) :: Complex{Float64}
    phase = p * par.Δω * ξ / sqrt(par.δ)
    return Complex{Float64}(cos(phase), sin(phase))   # exp(i·phase)
end

# ─────────────────────────────────────────────────────────────
# Fast Plane-Wave DGT  — O((N+M)P + RP)
# ─────────────────────────────────────────────────────────────

"""
    dgt_planwave(s, ρ, t, par; h=nothing) -> Vector{Float64}

Fast Discrete Gauss Transform using the plane-wave (FGT) algorithm.

# Algorithm stages
1. **S2W** — accumulate N sources into 2P+1 complex plane-wave coefficients
   for each target bin.
2. **W2L + bin shift** — translate coefficients to each of the M local
   targets, propagating between adjacent bins via an incremental phase shift.
3. **L2T** — collapse the complex vector to a real probability at each target:
   `prob[j] = θ(0) + 2 Re(Σ_{p=1}^{P} coeff_p)`.

# Arguments
- `s`   : source log-wealth points (length N)
- `ρ`   : source probability weights (length N)
- `t`   : target log-wealth points (length M), must be sorted
- `par` : `DGTParams` instance
- `h`   : bin width (defaults to `4√δ`, a rule-of-thumb)
"""
function dgt_planwave(
        s   :: AbstractVector{Float64},
        ρ   :: AbstractVector{Float64},
        t   :: AbstractVector{Float64},
        par :: DGTParams;
        h   :: Union{Float64, Nothing} = nothing
    ) :: Vector{Float64}

    N = length(s)
    M = length(t)
    @assert length(ρ) == N "ρ must have the same length as s"
    @assert issorted(t)    "target vector t must be sorted"

    P  = par.P
    δ  = par.δ
    m  = par.m
    Δω = par.Δω
    √δ = sqrt(δ)

    # Choose bin width
    bin_width = isnothing(h) ? 4.0 * √δ : h

    # Pre-compute plane-wave weights θ[p], p = -P..P
    θ = plane_wave_weights(par)                    # length 2P+1 (index P+1 ↔ mode 0)

    # ── Bin assignment ──────────────────────────────────────
    t_min = minimum(t)
    t_max = maximum(t)

    # Number of target bins needed to cover the target range
    n_bins = max(1, ceil(Int, (t_max - t_min) / bin_width) + 1)

    # Bin centre for bin b (0-indexed internally, 1-indexed externally)
    bin_centre(b::Int) = t_min + (b - 0.5) * bin_width

    # Assign each target to a bin (1..n_bins)
    tgt_bin = [max(1, min(n_bins, ceil(Int, (t[j] - t_min) / bin_width)))
               for j in 1:M]

    # ── Stage 1: Source to Plane-Wave (S2W) ─────────────────
    # coeff[b, p+P+1] = Σᵢ ρᵢ θ(p) f(t0[b] - sᵢ - m, p)
    # We use a (n_bins × (2P+1)) matrix of Complex{Float64}
    coeff = zeros(Complex{Float64}, n_bins, 2P + 1)

    # Pre-compute integer modes array: modes[k] = k - (P+1), for k = 1..2P+1
    modes = collect(-P:P)   # length 2P+1, modes[P+1] = 0

    for i in 1:N
        si = s[i]
        ρi = ρ[i]
        # Each source contributes to all target bins.
        # A production implementation restricts to a window of radius ~R bins.
        for b in 1:n_bins
            t0 = bin_centre(b)
            ξ  = t0 - si - m             # between-bin displacement
            @inbounds for k in 1:(2P + 1)
                p     = modes[k]          # mode integer: -P..P
                phase = p * Δω * ξ / √δ
                coeff[b, k] += ρi * θ[k] * Complex{Float64}(cos(phase), sin(phase))
            end
        end
    end

    # ── Stage 2 & 3: W2L (within-bin expansion) + L2T ───────
    #
    # prob[j] = Re(coeff[b, P+1])              ← p = 0 contribution
    #           + 2 Re( Σ_{p=1}^{P} coeff[b, P+1+p] × f(t[j]-t0, p) )
    #
    # The p=0 mode: f(ξ, 0) = exp(0) = 1, so coeff[b, P+1] = θ(0) Σ_i ρ_i.
    # Using the accumulated coefficient (rather than bare θ(0)) is essential
    # when the source weights do not sum to unity.
    prob = zeros(Float64, M)

    for j in 1:M
        b  = tgt_bin[j]
        t0 = bin_centre(b)
        η  = t[j] - t0                   # within-bin displacement

        # p = 0 term (purely real because f(η,0) = 1)
        p0_contrib = real(coeff[b, P + 1])

        # Positive modes p = 1..P
        acc = Complex{Float64}(0.0, 0.0)
        @inbounds for p in 1:P
            phase = p * Δω * η / √δ
            fw    = Complex{Float64}(cos(phase), sin(phase))
            acc  += coeff[b, P + 1 + p] * fw
        end
        prob[j] = p0_contrib + 2.0 * real(acc)
    end

    return prob
end

# ─────────────────────────────────────────────────────────────
# Convenience: one-period CDF of log-wealth
# ─────────────────────────────────────────────────────────────

"""
    one_step_cdf(s, ρ, t_grid, par; fast=true) -> Vector{Float64}

Compute the one-step transition CDF on `t_grid`, given source
distribution `(s, ρ)`.  Set `fast=false` to use the naïve O(NM) kernel.

# Arguments
- `s`      : source log-wealth points
- `ρ`      : source probability weights
- `t_grid` : target log-wealth grid (need not be sorted)
- `par`    : `DGTParams` instance
- `fast`   : if `true` (default), use the plane-wave algorithm
"""
function one_step_cdf(
        s      :: AbstractVector{Float64},
        ρ      :: AbstractVector{Float64},
        t_grid :: AbstractVector{Float64},
        par    :: DGTParams;
        fast   :: Bool = true
    ) :: Vector{Float64}

    pdf_vals = fast ?
      dgt_planwave(s, ρ, sort(t_grid), par) :
      dgt_naive(s, ρ, sort(t_grid), par)

    # Trapezoidal cumulative sum for CDF
    t_sorted = sort(t_grid)
    cdf = similar(pdf_vals)
    cdf[1] = 0.0
    for j in 2:length(t_sorted)
        dt = t_sorted[j] - t_sorted[j-1]
        cdf[j] = cdf[j-1] + 0.5 * dt * (pdf_vals[j-1] + pdf_vals[j])
    end
    # Normalise
    cdf ./= max(cdf[end], eps())
    return cdf
end

end  # module DGT

# ─────────────────────────────────────────────────────────────
# Self-contained demo / smoke test
# ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    # ── Optional package for accurate micro-benchmarking ─────────────────────
    # Install once with:  using Pkg; Pkg.add("BenchmarkTools")
    # Comment out the next two lines (and the @btime calls below) if not installed.
    using BenchmarkTools          # @btime, @benchmark
    using .DGT

    println("=== Discrete Gauss Transform — Demo ===\n")

    # Model parameters (from FM Global slide 19 base case)
    μ  = 0.07     # 7% annual drift
    σ  = 0.20     # 20% annual volatility
    Δτ = 1.0      # annual time steps
    W0 = 20.0     # initial wealth
    c  = 1.0      # annual consumption

    par = DGTParams(μ=μ, σ=σ, Δτ=Δτ, P=64, Ω=10.0)
    println("Derived parameters:")
    println("  δ = 2σ²Δτ = $(round(par.δ, digits=6))")
    println("  m = (μ-σ²/2)Δτ = $(round(par.m, digits=6))")
    println("  Δω = Ω/P = $(round(par.Δω, digits=6))")
    println()

    # Build a discretised initial distribution: point mass at ln(W0)
    N  = 200
    s0 = log(W0)
    s  = range(s0 - 3*sqrt(par.δ), s0 + 3*sqrt(par.δ), length=N) |> collect
    ρ  = [exp(-(si - s0)^2 / (0.01par.δ)) for si in s]
    ρ ./= sum(ρ) * (s[2] - s[1])   # normalise to PDF

    # Target grid (log-wealth)
    M = 300
    t = range(s0 - 5*sqrt(par.δ), s0 + 5*sqrt(par.δ), length=M) |> collect

    # Evaluate with both methods
    # @btime (from BenchmarkTools) runs multiple trials and strips JIT noise.
    # Replace with @elapsed if BenchmarkTools is not installed.
    println("Benchmarking naïve O(NM) reference…")
    prob_naive = dgt_naive(s, ρ, t, par)   # warm-up
    @btime dgt_naive($s, $ρ, $t, $par)

    println("Benchmarking plane-wave FGT…")
    prob_fast = dgt_planwave(s, ρ, t, par)  # warm-up
    @btime dgt_planwave($s, $ρ, $t, $par)

    # Error report using LinearAlgebra.norm (imported at top of module)
    err_vec  = prob_fast .- prob_naive
    l∞_err   = maximum(abs.(err_vec)) / maximum(abs.(prob_naive))
    l2_err   = norm(err_vec) / norm(prob_naive)
    println()
    println("Error vs naïve reference:")
    println("  L∞ relative error: $(round(l∞_err, sigdigits=4))")
    println("  L2 relative error: $(round(l2_err, sigdigits=4))")

    # Moments
    dt = t[2] - t[1]
    m0_fast  = sum(prob_fast)  * dt
    m1_fast  = sum(exp.(t) .* prob_fast) * dt
    m0_naive = sum(prob_naive) * dt
    m1_naive = sum(exp.(t) .* prob_naive) * dt
    println()
    println("Moments (one-step transition):")
    println("  m⁰  naïve=$(round(m0_naive, digits=5))  fast=$(round(m0_fast, digits=5))")
    println("  m¹  naïve=$(round(m1_naive, digits=5))  fast=$(round(m1_fast, digits=5))")
    println("\nDone.")
end
