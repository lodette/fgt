module FGTMonteCarlo

using Random
using Statistics

export simulate_wealth, terminal_wealth, mc_distribution, raw_moments

"""
    simulate_wealth(mu, sigma, consumption, W0;
                    nsim=100_000,
                    seed=nothing,
                    store_paths=false)

Monte Carlo simulation with period timing

    W[t+1] = (W[t] - c[t]) *
             exp(mu[t] - 0.5 * sigma[t]^2 + sigma[t] * Z[t])

where Z[t] are IID standard normal random variables.

W0 is wealth at the start of period 0. After T steps, where
T = length(consumption), the terminal result is wealth at the start of
period T, before period-T consumption.
"""
function simulate_wealth(mu, sigma,
                         consumption::AbstractVector{<:Real},
                         W0::Real;
                         nsim::Integer=100_000,
                         seed::Union{Nothing,Integer}=nothing,
                         store_paths::Bool=false)

    T = length(consumption)
    T > 0 || throw(ArgumentError("consumption must not be empty"))
    nsim > 0 || throw(ArgumentError("nsim must be positive"))

    μ = _expand_parameter(mu, T, "mu")
    σ = _expand_parameter(sigma, T, "sigma")

    any(x -> x < 0.0, σ) &&
        throw(ArgumentError("sigma must be nonnegative"))

    c = Float64.(consumption)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    w = fill(Float64(W0), nsim)

    if store_paths
        paths = Matrix{Float64}(undef, nsim, T + 1)
        paths[:, 1] .= w

        for t in 1:T
            z = randn(rng, nsim)
            @. w = (w - c[t]) *
                   exp(μ[t] - 0.5 * σ[t]^2 + σ[t] * z)
            paths[:, t + 1] .= w
        end

        return paths
    end

    for t in 1:T
        z = randn(rng, nsim)
        @. w = (w - c[t]) *
               exp(μ[t] - 0.5 * σ[t]^2 + σ[t] * z)
    end

    return w
end

terminal_wealth(mu, sigma, consumption, W0; kwargs...) =
    simulate_wealth(mu, sigma, consumption, W0;
                    store_paths=false, kwargs...)

function mc_distribution(mu, sigma,
                         consumption::AbstractVector{<:Real},
                         W0::Real;
                         nsim::Integer=100_000,
                         seed::Union{Nothing,Integer}=nothing,
                         nbins::Integer=200)

    nbins >= 2 || throw(ArgumentError("nbins must be at least 2"))

    terminal = terminal_wealth(mu, sigma, consumption, W0;
                               nsim=nsim, seed=seed)

    lo, hi = extrema(terminal)

    if lo == hi
        return [lo], [1.0], terminal
    end

    edges = collect(range(lo, hi; length=nbins + 1))
    counts = zeros(Int, nbins)
    scale = nbins / (hi - lo)

    @inbounds for x in terminal
        j = x == hi ? nbins : Int(floor((x - lo) * scale)) + 1
        j = clamp(j, 1, nbins)
        counts[j] += 1
    end

    wealth = @. 0.5 * (edges[1:end-1] + edges[2:end])
    mass = counts ./ nsim

    return wealth, mass, terminal
end

function raw_moments(wealth::AbstractVector{<:Real}; orders=1:4)
    return [mean(x -> x^k, wealth) for k in orders]
end

function _expand_parameter(x::Real, T::Int, name::AbstractString)
    return fill(Float64(x), T)
end

function _expand_parameter(x::AbstractVector{<:Real},
                           T::Int,
                           name::AbstractString)
    length(x) >= T ||
        throw(DimensionMismatch("$name is shorter than consumption"))
    return Float64.(x[1:T])
end

end # module
