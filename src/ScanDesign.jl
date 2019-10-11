module ScanDesign

using NLopt, Distributions
using LinearAlgebra, Statistics

export scandesign
export fisher
export expectedcost

include("fisher.jl")
include("expectedcost.jl")
include("design.jl")

end
