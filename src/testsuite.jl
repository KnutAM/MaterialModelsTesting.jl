@kwdef struct LoadInfo{T}
    ϵsmall::T = 1e-4 # Strain at low loading (e.g. elastic, single step)
    ϵlarge::T = 1e-2 # Strain at high loading (e.g. plastic), use `nsteps`
    nsteps::Int = 10 # Number of steps to reach `ϵlarge`
    Δt::T = 1.0      # Time step during loading
end

"""
    test_material(m::AbstractMaterial, [loadinfo::LoadInfo];
        basics = true, conversions = false, differentiation = false
        )

Function to quickly test that the material `m` follows the `MaterialModelsBase`
interface, and that what generally should hold true, does. All material models
should pass the `basics` tests, which are on by default, whereas the `conversions`
and `differentiation` tests are optional.

`loadinfo` gives information about how much the material is loaded, and different values
can be supplied if required for the specific material model/parameters. 

## `basics` (default: `true`)
Tests that required functions - `initial_material_state` and `allocate_material_cache`
returns the correct types, and runs the `material_response` to check that the outputs
are of correct type. Furthermore, it checks that the tangent stiffness matches with the
numerical derivative, although this test is only approximate.

## `conversions` (default: `false`)
Tests the conversions `tovector`, `tovector!`, `fromvector` for both the material parameters
and the state variables, along with the associated methods `get_num_params`, `get_num_statevars`,
`get_params_eltype`, and `get_statevar_eltype`. Furthermore, it is tested that `initial_material_state`
returns a state using the element type from the material, and that the conversions respect the different
element types of the given vector. Since also `Float32` is used in this testing, it is important not to 
default to using `Float64` when creating e.g. state variables, i.e. use `zero(Tensor{2,3,T})` instead 
of `zeros(Tensor{2,3})`.

## `differentiation` (default: `false`)
Tests that the `differentiate_material!` function works correctly. Uses `BigFloat` to calculate the 
numerical derivatives to avoid floating point errors.
"""
function test_material(m::AbstractMaterial, loadinfo = LoadInfo();
        basics = true, conversions = false, differentiation = false
        )
    basics && test_material_basic(m, loadinfo)
    conversions && test_material_conversions(m, loadinfo)
    differentiation && test_material_differentiation(m, loadinfo)
    return nothing
end

function get_strain(::Type{<:Tensor{2, dim}}, ϵval) where {dim}
    I2 = one(Tensor{2, dim})
    return I2 + normalize(rand(Tensor{2, dim})) * ϵval # Finite strain => F
end
function get_strain(::Type{<:SymmetricTensor{2, dim}}, ϵval) where {dim}
    return normalize(rand(SymmetricTensor{2, dim})) * ϵval # Small strain => ϵ
end

function test_material_basic(args...)
    @testset "Basic tests" begin
        _test_material_basic(args...)
    end
    return nothing
end
function _test_material_basic(m::AbstractMaterial, loadinfo)
    # Extract some basic data to use later
    (;Δt, ϵsmall, ϵlarge, nsteps) = loadinfo
    TB = MMB.get_tensorbase(m)
    ϵ = get_strain(TB, ϵsmall)

    # Check that required interface functions works
    @test isa(initial_material_state(m), AbstractMaterialState)
    @test isa(allocate_material_cache(m), AbstractMaterialCache)
    
    # Check that the `material_response` call returns the correct type
    old_state = initial_material_state(m)
    cache = allocate_material_cache(m)
    extras = NoExtraOutput()
    σ, dσdϵ, new_state = material_response(m, ϵ, old_state, Δt, cache, extras)
    @test typeof(ϵ) == typeof(σ)
    @test typeof(dσdϵ) == typeof(ϵ ⊗ ϵ)
    @test typeof(new_state) == typeof(old_state)

    # Check dσdϵ
    # This can be checked more carefully in test_material_differentiation
    # as that requires that the conversion routines are implemented, allowing
    # differentiation wrt. BigFloat...
    #  Initial loading
    sf(ev) = tomandel(material_response(m, frommandel(TB, ev), old_state, Δt, cache, extras)[1])
    relstep = 1e-6
    dσdϵ_num = FiniteDiff.finite_difference_jacobian(sf, tomandel(ϵ), Val{:central}; relstep)
    compare_derivatives(tomandel(dσdϵ), dσdϵ_num, tomandel(σ), tomandel(ϵ) * relstep;
        atol_min = eps(), rtol_min = eps())
    #  Load stepping
    Δϵ = (get_strain(TB, ϵlarge) - get_strain(TB, 0.0)) / nsteps
    for i in 1:nsteps
        ϵ = get_strain(TB, 0.0) + Δϵ * i
        σ, dσdϵ, new_state = material_response(m, ϵ, old_state, Δt, cache, extras)
        σf(ev) = tomandel(material_response(m, frommandel(TB, ev), old_state, Δt, cache, extras)[1])
        dσdϵ_num = FiniteDiff.finite_difference_jacobian(σf, tomandel(ϵ), Val{:central}; relstep)
        compare_derivatives(tomandel(dσdϵ), dσdϵ_num, tomandel(σ), tomandel(ϵ) * relstep;
            atol_min = eps(), rtol_min = eps())
        old_state = new_state
    end
end

function test_vectorconversion(::Type{T}, obj, num_items) where {T}
    x0 = [rand(T) for _ in 1:num_items]
    obj1 = fromvector(x0, obj)
    @test eltype(tovector(obj1)) == T  # Conversion respects eltype of x0
    x1 = similar(x0)
    @test x1 === tovector!(x1, obj1) # Returns mutated vector
    @test x1 ≈ x0                    # Values correctly updated
    
    # Test interface with nondefault offset
    x0 = [rand(T) for _ in 1:(num_items + 2)]
    obj1 = fromvector(x0, obj; offset = 1)
    @test eltype(tovector(obj1)) == T  # Conversion respects eltype of x0
    x1 = similar(x0)
    a, b = rand(T, 2)
    x1[1] = a; x1[end] = b;
    @test x1 === tovector!(x1, obj1; offset = 1) # Returns mutated vector
    @test x1[2:end-1] ≈ x0[2:end-1]              # Values correctly updated
    @test a == x1[1]    # First value not touched
    @test b == x1[end]  # Last value not touched
end


function test_material_conversions(args...)
    @testset "Conversion tests" begin
        _test_material_conversions(args...)
    end
    return nothing
end
function _test_material_conversions(m0::AbstractMaterial, loadinfo)
    v0 = tovector(m0)
    s0 = initial_material_state(m0)
    @test length(v0) == MMB.get_num_params(m0)
    @test length(tovector(s0)) == MMB.get_num_statevars(m0)
    
    @testset for T in (Float32, Float64, BigFloat)
        v1 = [rand(T) for _ in 1:MMB.get_num_params(m0)]
        m1 = fromvector(v1, m0)
        @testset "material parameters" begin
            @test MMB.get_params_eltype(m1) == T
            test_vectorconversion(T, m0, MMB.get_num_params(m0))
        end
        @testset "state variables" begin
            state = initial_material_state(m1)
            @test eltype(tovector(state)) == T # Type taken from material type
            test_vectorconversion(T, s0, MMB.get_num_statevars(m0))
        end
    end
    return nothing
end

function test_material_differentiation(args...)
    @testset "Differentiation tests" begin
        _test_material_differentiation(args...)
    end
    return nothing
end
function _test_material_differentiation(m::AbstractMaterial, loadinfo)
    @info "test_material_differentiation not implemented"
end