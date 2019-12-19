"""
    scandesign(P0, costfun; algorithm, lb, ub, stopval, tol_rel, tol_abs,
               maxeval, maxtime, localalgorithm, verbosecount)

Design an optimal scan profile.

# Arguments
- `P0::Real`: Initial scan parameters [numScanTypes][numScanParams[n]][D[n]],
  where numScanParams[n] is the number of scan parameters for the nth scan
  type, and D[n] is the number of scans of the nth scan type
- `costfun::Function`: Cost function to minimize, has calling signature
  `(cost [, grad]) = costfun(P, computegrad)` where `P` is the scan design and
  `computegrad` is a `Bool` that specifies whether to compute the gradient of
  the cost function
- `algorithm::Symbol = :G_MLSL`: The algorithm to use for the optimization
- `lb::Real = -Inf`: Lower bounds on scan parameters (same shape as P0)
- `ub::Real = Inf`: Upper bounds on scan parameters (same shape as P0)
- `stopval::Real = -Inf`: Target cost function value
- `tol_rel::Real = 0`: Relative tolerance for cost function value
- `tol_abs::Real = 0`: Absolute tolerance for cost function value
- `maxeval::Integer = 0`: Maximum number of function evaluations (0 or
  negative means there is no constraint)
- `maxtime::Real = 0`: Maximum amount of time to run the optimization
  algorithm for, in seconds (0 or negative means there is no constraint)
- `localalgorithm::Symbol = :LD_SLSQP`: The algorithm to use for the local
  optimization for the main global algorithm; only used for some values of
  `algorithm`
- `verbosecount::Real = 100`: Number of function calls between each
  informative print statement; set to Inf to disable print statements

# Return
- `P::Real`: Optimal scan parameters [numScanTypes][numScanParams[n]][D[n]]
- `cost::Real`: Cost at the optimal point
- `flag::Symbol`: Return code, see https://nlopt.readthedocs.io/en/latest/
  NLopt_Reference/#Return_values
"""
function scandesign(
    P0::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real,1},1},1},
    costfun::Function;
    algorithm::Symbol = :G_MLSL,
    lb::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real,1},1},1} =
        [[[-Inf for d in 1:length(P0[n][p])] for p in 1:length(P0[n])]
        for n in 1:length(P0)],
    ub::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real,1},1},1} =
        [[[Inf for d in 1:length(P0[n][p])] for p in 1:length(P0[n])]
        for n in 1:length(P0)],
    stopval::Real = -Inf,
    tol_rel::Real = 0.0,
    tol_abs::Real = 0.0,
    maxeval::Integer = 0,
    maxtime::Real = 0.0,
    localalgorithm::Symbol = :LD_SLSQP,
    verbosecount::Real = 100
)

    # Grab constants
    numScanTypes = length(P0)
    numScanParams = [length(P0[n]) for n in 1:numScanTypes]
    D = [length(P0[n][1]) for n in 1:numScanTypes]

    # Create function for undoing P2Pv (to avoid passing in the above constants
    # again and again)
    PfromPv = Pv -> Pv2P(Pv, numScanTypes, numScanParams, D)

    # Convert P0, lb, ub into vectors
    P0v = P2Pv(P0) # [sum(numScanParams .* D)]
    lbv = P2Pv(lb) # This works because P0 and lb and ub have the same shape
    ubv = P2Pv(ub)

    # Set up the cost function
    fn = createcostfunction(costfun, PfromPv, verbosecount)

    # Set up the optimization problem
    problem = Opt(algorithm, length(P0v))
    min_objective!(problem, fn)
    lower_bounds!(problem, lbv)
    upper_bounds!(problem, ubv)
    stopval!(problem, stopval)
    ftol_rel!(problem, tol_rel)
    ftol_abs!(problem, tol_abs)
    maxeval!(problem, maxeval)
    maxtime!(problem, maxtime)

    # Set up the local optimization problem; this won't be used for some values
    # of `algorithm`, but it doesn't seem to hurt anything
    localprob = Opt(localalgorithm, length(P0v))
    local_optimizer!(problem, localprob)

    # Optimize the scan design
    (cost, Pv, flag) = optimize(problem, P0v)

    # Convert Pv back to P to return
    P = PfromPv(Pv)

    # Print something to indicate that the optimization finished
    println("Optimization finished in $(calls::Integer) function call(s) " *
            "with exit code $flag.")

    return (P, cost, flag)

end

"Convert the scan parameters into a vector. P is originally
[numScanTypes][numScanParams[n]][D[n]]; Pv will be
[sum_n(numScanParams[n] * D[n])]. Looking at P[n][p][d] and Pv[i], as i
increases, first d increases, then p, then n."
function P2Pv(P)

    Pv = cat(dims = 1, [cat(dims = 1, P[n]...) for n in 1:length(P)]...)

end

"Convert the vectorized scan parameters back into its original format."
function Pv2P(Pv, numScanTypes, numScanParams, D)

    P = [[Pv[(sum(numScanParams[1:n-1] .* D[1:n-1]) + (p-1) * D[n] + 1):(
              sum(numScanParams[1:n-1] .* D[1:n-1]) + p * D[n])]
          for p in 1:numScanParams[n]] for n in 1:numScanTypes]

end

"Create the cost function to be minimized."
function createcostfunction(costfun, PfromPv, verbosecount)

    # Keep track of the number of times the cost function is evaluated
    global calls
    calls = 0

    # Create the cost function
    fn(Pv::Vector, gradv::Vector) = begin

        # Keep track of function calls
        global calls
        calls::Integer += 1

        # Convert Pv to a scan design to pass to expected_cost(...)
        P = PfromPv(Pv)

        # Only compute the gradient if gradv isn't empty
        if length(gradv) > 0

            # I was going to explicitly check if gradxp is empty, but this will
            # throw an error already if gradxp is empty
            (cost, grad) = costfun(P, true)
            gradv[:] = P2Pv(grad) # Works because grad and P have same shape

        else

            cost = costfun(P, false)

        end

        # Print every verbose_count calls
        if calls::Integer % verbosecount == 0
            println("Call count = $(calls::Integer), cost = $cost")
        end

        return cost

    end

    return fn

end
