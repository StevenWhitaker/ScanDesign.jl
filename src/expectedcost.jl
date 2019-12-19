"""
    expectedcost(P, Σ, xPriors, νPriors, gradx, gradxp; computegrad)

Calculate the expected cost of the scan profile. If `computegrad` is true, also
calculate the gradient of the expected cost.

# Arguments
- `P::Real`: Scan parameters [numScanTypes][numScanParams[n]][D[n]], where
  numScanParams[n] is the number of scan parameters for the nth scan type,
  and D[n] is the number of scans of the nth scan type
- `Σ::Real`: Scan noise standard deviation [numScanTypes][D[n]], where D[n] is
  the number of scans of the nth scan type
- `xPriors::Dict{String,Any}`: Information about latent parameter prior
  distributions [L]; more information below
- `νPriors::Dict{String,Any}`: Information about known parameter prior
  distributions [K]; more information below
- `gradx::Function`: Gradients of the signal models with respect to the latent
  parameters [numScanTypes]
- `gradxp::Function`: Gradients of the signal models with respect to
  the latent parameters and scan parameters [numScanTypes]
- `computegrad::Bool = false`: Whether or not to compute the gradient

## Priors
Each element of `xPriors` and `νPriors` should have the folowing entries.

- `"dist"::Distribution`: Distribution from which to draw samples
- `"nsamp"::Integer`: Number of samples to draw
- `"cfvar"::Bool`: Whether to weight the distribution to account for relative
  deviations from the parameter value; is it preferable to be within x of
  the true value (false), or is it preferable to be within x% of the true
  value (true)?

Additionally, each element of `xPriors` should also have the following entry.

- `"weight"::Real`: Relative importance of accurately estimating the parameter

# Return
- `cost::Real`: Expected cost of the scan profile
- `grad::Real`: Gradient of the expected cost function
  [numScanTypes][numScanParams[n]][D[n]]
"""
function expectedcost(
    P::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real,1},1},1},
    Σ::AbstractArray{<:AbstractArray{<:Real,1},1},
    xPriors::AbstractArray{Dict{String,Any},1},
    νPriors::AbstractArray{Dict{String,Any},1},
    gradx::AbstractArray{<:Function,1},
    gradxp::AbstractArray{<:Function,1};
    computegrad::Bool = false
)

    # Grab the number of latent and known parameters
    L = length(xPriors)
    K = length(νPriors)

    # Make sure all the weights are nonnegative
    if any([xPriors[i]["weight"] < 0 for i in 1:L])
        error("Negative weights detected. Only nonnegative weights allowed.")
    end

    # Design the latent/known marginal distributions
    priors = createmarginaldists!(xPriors, νPriors, L, K)

    # Grab the number of samples for each parameter
    nsamps = [priors[i]["nsamp"] for i in 1:L+K]

    # Construct the joint distribution from the marginals
    joint = createjointdist([priors[i]["prob"] for i in 1:L+K], nsamps)
                                                                  # [nsamps...]

    # Make sure the joint distribution is normalized
    if abs(sum(joint[:]) - 1) > eps() * length(joint)
        error("Joint distribution isn't normalized?")
    end

    # Create all combinations of the latent and known parameters associated with
    # each entry in joint
    samples = createsamplepoints(priors, nsamps) # [L+K][N]

    # Prepare the diagonal weighting matrix
    totalWeight = sum([xPriors[i]["weight"] for i in 1:L])
    W = diagm(0 => [xPriors[i]["weight"] / totalWeight for i in 1:L]) # [L,L]

    # Calculate the Fisher information matrices for each set of sample points
    F = fisher(gradx, samples[1:L], samples[L+1:end], P, Σ) # [N,L,L]

    # Calculate the weighted precision for each set of sample points
    # wtprec = [trace(W * (F[n,:,:] \ (W'))) for n in 1:prod(nsamps)] # [N]
    # wtprec = [trace(W * pinv(F[n,:,:]) * (W')) for n in 1:prod(nsamps)] # [N]
    wtprec = [try tr(W * (F[n,:,:] \ (W'))) catch ex
            ex isa SingularException ? tr(W * pinv(F[n,:,:]) * (W')) : throw(ex)
            end for n in 1:prod(nsamps)] # [N]

    # Calculate the expected cost, which integrates wtprec over the distribution
    cost = abs(sum(wtprec .* joint[:]))

    # See if the gradient should be computed; if not, return the cost
    if !computegrad
        return cost
    end

    # Calculate the gradient of the expected cost with respect to the scan
    # parameters
    # M = [W / F[n,:,:] for n in 1:prod(nsamps)] # [N][L,L]
    # M = [W * pinv(F[n,:,:]) for n in 1:prod(nsamps)] # [N][L,L]
    M = [try W / F[n,:,:] catch ex ex isa SingularException ?
         W * pinv(F[n,:,:]) : throw(ex) end for n in 1:prod(nsamps)] # [N][L,L]
    grad = expectedcostgrad(M, samples[1:L], samples[L+1:end], P, Σ, gradx,
                            gradxp, joint)

    return (cost, grad)

end

"Create marginal distributions of the samples of the prior distributions."
function createmarginaldists!(xPriors, νPriors, L, K)

    # Combine xPriors and νPriors for easy iteration
    priors = cat(dims = 1, xPriors, νPriors)

    # Design the latent/known parameter distributions
    for i = 1:L+K

        # Specify the minimum and maximum values to use
        priors[i]["min"] = priors[i]["nsamp"] == 1 ? mean(priors[i]["dist"]) :
                                                     minimum(priors[i]["dist"])
        if isinf(priors[i]["min"])
            # Sample 1-3 standard deviations from the mean, depending on nsamp
            nstd = min(floor(priors[i]["nsamp"] / 2), 3)
            priors[i]["min"] = mean(priors[i]["dist"]) - nstd*std(priors[i]["dist"])
        end

        priors[i]["max"] = priors[i]["nsamp"] == 1 ? mean(priors[i]["dist"]) :
                                                     maximum(priors[i]["dist"])
        if isinf(priors[i]["max"])
            # Sample 1-3 standard deviations from the mean, depending on nsamp
            nstd = min(floor(priors[i]["nsamp"] / 2), 3)
            priors[i]["max"] = mean(priors[i]["dist"]) + nstd*std(priors[i]["dist"])
        end

        # Get the values to be used from the distribution
        priors[i]["val"] = LinRange(priors[i]["min"], priors[i]["max"],
                                    priors[i]["nsamp"])

        # Set the probabilities of the new distribution of the given samples
        priors[i]["prob"] = pdf.(priors[i]["dist"], priors[i]["val"])

        # Compensate if using coefficients of variation
        if priors[i]["cfvar"]
            priors[i]["prob"] ./= max.(abs.(priors[i]["val"]).^2, eps())
        end

        # Normalize the probabilities
        priors[i]["prob"] = safedivide.(priors[i]["prob"],
                                        norm(priors[i]["prob"], 1))

    end

    return priors

end

"Create a joint distribution as the product of marginals."
function createjointdist(marginals, nsamps)

    joint = marginals[1]
    for i = 2:length(marginals)
        joint = kron(marginals[i], joint)
    end
    joint = reshape(joint, nsamps...)

    return joint

end

"Create the samples points of the joint distribution."
function createsamplepoints(priors, nsamps)

    # Initialize the samples
    # I don't actually need to initialize anything because I can use a list
    # comprehension
    # samples = Array{Array{Float64,1},1}(length(nsamps))

    # The first parameter varies the quickest, so just repeat it however many
    # times are needed so that length(samples[1]) == prod(nsamps)
    # I would normally think to do the following, but the code for the middle
    # parameters generalizes because nsamps[1:0] is a length-0 array, and
    # prod(...) return 1 when given a length-0 array
    # samples[1] = repmat(priors[1]["val"], prod(nsamps[2:end]))

    # The middle parameters vary slower than the first but more quickly than the
    # last, so repeat their transposes to duplicate each entry, then repeat the
    # resulting pattern
    # I actually realized that the for loop I would normally write can be
    # expressed using a list comprehension
    # for i = 1:length(nsamps)
    #   samples[i] = repmat(repmat(priors[i]["val"].', prod(nsamps[1:i-1]))[:],
    #                       prod(nsamps[i+1:end]))
    # end

    # The last parameter varies the slowest, so repeat each individual entry
    # enough times before going to the next entry
    # I would normally think to do the following, but the code for the middle
    # parameters generalizes because nsamps[end+1:end] is a length-0 array, and
    # prod(...) return 1 when given a length-0 array
    # samples[end] = repmat(priors[end]["val"].', prod(nsamps[1:end-1]))

    # It turns out I can do all of the above with one list comprehension
    # statement
    samples = [repeat(repeat(transpose(priors[i]["val"]), prod(nsamps[1:i-1]))[:],
                    prod(nsamps[i+1:end])) for i in 1:length(nsamps)]

    return samples

end

function expectedcostgrad(M, x, ν, P, Σ, gradx, gradxp, joint)

    # Initialize grad
    grad = Array{Array{Array{Float64,1},1},1}(undef, length(gradx))
                                       # [numScanTypes][numScanParams[n]][D[n]]

    # Loop through each scan type
    for scanType = 1:length(gradx)

        # Evaluate the given gradients
        gx  = gradx[scanType](x..., ν..., [transpose(p) for p in P[scanType]]...)
                                                                  # [L][N,D[n]]
        gxp = gradxp[scanType](x..., ν..., [transpose(p) for p in P[scanType]]...)
                                                 # [L,numScanParams[n]][N,D[n]]
        gx  = cat(dims = 3, gx...) # [N,D[n],L]
        gxp = cat(dims = 4, [cat(dims = 3, gxp[:,p]...) for p in 1:size(gxp,2)]...)
                                                  # [N,D[n],L,numScanParams[n]]

        # Calculate the cost gradients for the current scan type
        g = [[-2 * real(tr(M[n] * safedivide.(gx[n,d,:] * gxp[n,d,:,p]',
              Σ[scanType][d]^2) * M[n]')) for n in 1:length(x[1]),
              d in 1:length(Σ[scanType])] for p in 1:length(P[scanType])]
                                                   # [numScanParams[n]][N,D[n]]

        # Calculate the expected cost by integrating over joint
        grad[scanType] = [sum(g[p] .* joint[:], dims = 1)[:] for
                          p in 1:length(P[scanType])] # [numScanParams[n]][D[n]]

    end

    return grad

end
