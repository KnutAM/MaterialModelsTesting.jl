module MaterialModelsTesting
using MaterialModelsBase
using Test
using Tensors, StaticArrays, LinearAlgebra
using ForwardDiff, FiniteDiff

const DualT = ForwardDiff.Dual{ForwardDiff.Tag{:MatTest, T}, T, 2} where {T}

# Overloaded functions
import MaterialModelsBase as MMB

include("numdiff.jl")
include("loadcases.jl")
include("testmaterials/LinearElastic.jl")
include("testmaterials/NeoHooke.jl")
include("testmaterials/ViscoElastic.jl")
include("testsuite.jl")

end
