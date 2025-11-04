# Test material implementations
# * LinearElastic
# * ViscoElastic
# TODO: NeoHookean (geometrically nonlinear materials)

# LinearElastic material 
struct LinearElastic{T} <: AbstractMaterial 
    G::T
    K::T
end
function get_stiffness(m::LinearElastic{T}) where {T}
    I2 = one(SymmetricTensor{2, 3, T})
    Ivol = I2 ⊗ I2
    Isymdev = minorsymmetric(otimesu(I2, I2) - Ivol / 3)
    return 2 * m.G * Isymdev + m.K * Ivol
end
MMB.get_vector_eltype(::LinearElastic{T}) where {T} = T

function MMB.material_response(
        m::LinearElastic, ϵ::SymmetricTensor{2},
        old::NoMaterialState, 
        Δt, cache, extras) # Explicit values added to test the defaults
    dσdϵ = get_stiffness(m)
    σ = dσdϵ⊡ϵ
    return σ, dσdϵ, old
end

MMB.get_vector_length(::LinearElastic) = 2
function MMB.tovector!(v::Vector, m::LinearElastic; offset = 0)
    v[1 + offset] = m.G
    v[2 + offset] = m.K
    return v
end
MMB.fromvector(v::Vector, ::LinearElastic; offset = 0) = LinearElastic(v[1 + offset], v[2 + offset])

function MMB.differentiate_material!(
    diff::MaterialDerivatives,
    m::LinearElastic,
    ϵ::SymmetricTensor,
    args...)
    tomandel!(diff.dσdϵ, get_stiffness(m))

    σ_from_p(p::Vector) = tomandel(get_stiffness(fromvector(p, m))⊡ϵ)
    ForwardDiff.jacobian!(diff.dσdp, σ_from_p, tovector(m))
end
