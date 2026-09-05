module FGTMonteCarlo

using Random
using Statistics

export simulate_wealth, terminal_wealth, mc_distribution, raw_moments

"""
    simulate_wealth(mu, sigma, consumption, W0;
                    nsim=100_000,
                    seed=nothing,
                    store_paths=false)

Monte Carlo simulation of the wealth dynamics used by the current FGT runner.

Timing convention

    W[1] = W0 - consumption[1]

and, for t = 1, ..., T-1,

    W[t+1] = W[t] * exp(mu[t] - sigma[t]^2 / 2 + sigma[t] * Z[t])
             - consumption[t+1]

where Z[t] ~ N(0,1), independently across paths and periods.

This matches the current Julia FGT runner convention. The first consumption is
taken immediately, before the first stochastic return.

`mu` and `sigma` may each be either a scalar or a vector. If a vector is
supplied, it must contain at least `length(consumption)` elements.

Negative wealth is allowed. Since the gross return is always positive, a
negative wealth value remains negative after return multiplication unless a
subsequent cash flow changes its sign.

Returns

If `store_paths=false`, returns a vector containing terminal wealth for all
simulations.

If `store_paths=true`, returns an `nsim × T` matrix. Column 1 is wealth just
after the initial consumption. Column t is wealth after the consumption at
time t.
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
    any(<(0.0), σ) && throw(ArgumentError("sigma must be nonnegative"))

    c = Float64.(consumption)

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    w = fill(Float64(W0) - c[1], nsim)

    if store_paths
        paths = Matrix{Float64}(undef, nsim, T)
        paths[:, 1] .= w

        for t in 1:(T - 1)
            z = randn(rng, nsim)
            @. w = w * exp(μ[t] - 0.5 * σ[t]^2 + σ[t] * z) - c[t + 1]
            paths[:, t + 1] .= w
        end

        return paths
    end

    for t in 1:(T - 1)
        z = randn(rng, nsim)
        @. w = w * exp(μ[t] - 0.5 * σ[t]^2 + σ[t] * z) - c[t + 1]
    end

    return w
end


"""
    terminal_wealth(mu, sigma, consumption, W0; kwargs...)

Convenience alias for `simulate_wealth(...; store_paths=false)`.
"""
terminal_wealth(mu, sigma, consumption, W0; kwargs...) =
    simulate_wealth(mu, sigma, consumption, W0;
                    store_paths=false, kwargs...)


"""
    mc_distribution(mu, sigma, consumption, W0;
                    nsim=100_000,
                    seed=nothing,
                    nbins=200)

Simulate terminal wealth and return a histogram representation suitable for
comparison with `FGTrunner.distribution`.

Returns `(wealth, mass, terminal)`.

`wealth` contains histogram-bin centers.
`mass` contains probability mass in each bin and sums to approximately 1.
`terminal` contains the unbinned simulated terminal wealth values.
"""
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


"""
    raw_moments(wealth; orders=1:4)

Compute raw, non-centered Monte Carlo moments directly from simulated wealth.

For example,

    m = raw_moments(terminal)

returns E[W], E[W^2], E[W^3], and E[W^4].
"""
function raw_moments(wealth::AbstractVector{<:Real};
                     orders=1:4)
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
