module MaterialModelsTesting
using MaterialModelsBase
using Test
using Tensors, StaticArrays, LinearAlgebra
using ForwardDiff, FiniteDiff

# Overloaded functions
import MaterialModelsBase as MMB

include("numdiff.jl")
include("loadcases.jl")
include("testmaterials/LinearElastic.jl")
include("testmaterials/ViscoElastic.jl")
include("testsuite.jl")

end
