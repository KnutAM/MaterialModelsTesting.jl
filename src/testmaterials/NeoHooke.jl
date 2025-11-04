@kwdef struct NeoHooke{T} <: AbstractMaterial
    G::T
    K::T
end
function MMB.material_response(m::NeoHooke, F::Tensor{2,3}, old, args...)
    ∂P∂F, P, _ = hessian(defgrad -> free_energy(m, defgrad), F, :all)
    return P, ∂P∂F, old
end
function free_energy(m::NeoHooke, F::Tensor{2,3})
    C = tdot(F)
    detC = det(C)
    ΨG = (m.G / 2) * (tr(C) / cbrt(detC) - 3)
    ΨK = (m.K / 2) * (sqrt(detC) - 1)^2
    return ΨG + ΨK
end

MMB.get_vector_eltype(::NeoHooke{T}) where {T} = T
MMB.get_vector_length(::NeoHooke) = 2
MMB.get_tensorbase(::NeoHooke) = Tensor{2,3}

function MMB.tovector!(v::Vector, m::NeoHooke; offset = 0)
    v[1 + offset] = m.G
    v[2 + offset] = m.K
    return v
end
MMB.fromvector(v::Vector, ::NeoHooke; offset = 0) = NeoHooke(v[1 + offset], v[2 + offset])

# diff, m, ϵ, state, Δt, cache, extras, dσdϵ
function MMB.differentiate_material!(
    diff::MaterialDerivatives,
    m::NeoHooke,
    F::Tensor,
    args...)
    tomandel!(diff.dσdϵ, hessian(defgrad -> free_energy(m, defgrad), F))
    function P_from_params(params::Vector)
        m_dual = fromvector(params, m)
        P = gradient(defgrad -> free_energy(m_dual, defgrad), F)
        return tovector(P)
    end
    ForwardDiff.jacobian!(diff.dσdp, P_from_params, tovector(m))
end
