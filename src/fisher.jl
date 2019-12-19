"""
    fisher(grads, x, ν, P, Σ)

Calculate the Fisher information matrix.

# Arguments
- `grads::Function`: Gradients of the signal models with respect to the latent
  parameters [numScanTypes]
- `x::Real`: Latent parameters [L][N]
- `ν::Real`: Known parameters [K][N]
- `P::Real`: Scan parameters [numScanTypes][numScanParams[n]][D[n]], where
  numScanParams[n] is the number of scan parameters for the nth scan type,
  and D[n] is the number of scans of the nth scan type
- `Σ::Real`: Scan noise standard deviation [numScanTypes][D[n]], where D[n] is
  the number of scans of the nth scan type

# Return
- `F::Real`: Fisher information matrix for each set of latent/known parameters
  [N,L,L]
"""
function fisher(
    grads::AbstractArray{<:Function,1},
    x::AbstractArray{<:AbstractArray{<:Real,1},1},
    ν::AbstractArray{<:AbstractArray{<:Real,1},1},
    P::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real,1},1},1},
    Σ::AbstractArray{<:AbstractArray{<:Real,1},1}
)

    # Initialize the Fisher information matrices
    F = zeros(length(grads), length(x[1]), length(x), length(x))
                                                         # [numScanTypes,N,L,L]

    # Calculate the Fisher information matrices for each scan type
    for scanType = 1:length(grads)

        # Construct precision (inverse covariance) matrix
        prec = diagm(0 => safedivide.(1, Σ[scanType].^2)) # [D[n],D[n]]

        # Evaluate the gradient for each scan
        grad = grads[scanType](x..., ν..., [transpose(p) for p in P[scanType]]...)
                                                                  # [L][N,D[n]]
        grad = cat(dims = 3, grad...) # [N,D[n],L]

        # Calculate the Fisher information matrix for each set of latent/known
        # parameters
        for n = 1:length(x[1])

            F[scanType,n,:,:] = real(grad[n,:,:]' * prec * grad[n,:,:])

        end

    end

    # Add all the Fisher information matrices from each scan type
    return dropdims(sum(F, dims = 1), dims = 1) # [N,L,L]

end

"Protect against divide-by-zero errors by returning 0 if divisor is 0."
function safedivide(dividend::Number, divisor::Number)

    return divisor == 0 ? 0 : dividend / divisor

end
