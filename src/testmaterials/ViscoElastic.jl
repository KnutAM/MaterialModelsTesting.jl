# ViscoElastic material: Note - not physically reasonable model because 
# the volumetric response is also viscous
struct ViscoElastic{T} <: AbstractMaterial
    E1::LinearElastic{T}
    E2::LinearElastic{T}
    η::T
end
MMB.get_params_eltype(::ViscoElastic{T}) where {T} = T

function MMB.get_num_params(m::ViscoElastic)
    return sum(get_num_params.((m.E1, m.E2))) + 1
end

function MMB.tovector!(v::AbstractVector, m::ViscoElastic; offset = 0)
    i = offset
    tovector!(v, m.E1; offset = i); i += get_num_params(m.E1)
    tovector!(v, m.E2; offset = i); i += get_num_params(m.E2)
    v[i + 1] = m.η
    return v
end

function MMB.fromvector(v::AbstractVector, m::ViscoElastic; offset = 0)
    i = offset
    E1 = fromvector(v, m.E1; offset = i); i += get_num_params(m.E1)
    E2 = fromvector(v, m.E2; offset = i); i += get_num_params(m.E2)
    η = v[i + 1]
    return ViscoElastic(E1, E2, η)
end

struct ViscoElasticState{T} <: AbstractMaterialState
    ϵv::SymmetricTensor{2,3,T,6}
end
function MMB.initial_material_state(::ViscoElastic{T}) where {T}
    return ViscoElasticState(zero(SymmetricTensor{2, 3, T}))
end

MMB.get_num_statevars(s::ViscoElasticState) = 6
MMB.get_statevar_eltype(::ViscoElasticState{T}) where {T} = T
MMB.tovector!(v, s::ViscoElasticState; offset = 0) = tovector!(v, s.ϵv; offset)
MMB.fromvector(v, s::ViscoElasticState; offset = 0) = ViscoElasticState(fromvector(v, s.ϵv; offset))

# To allow easy automatic differentiation
function get_stress_ϵv(m::ViscoElastic, ϵ::SymmetricTensor{2,3}, old::ViscoElasticState, Δt)
    E1 = get_stiffness(m.E1)
    E2 = get_stiffness(m.E2)
    ϵv = inv(Δt*E2+one(E2)*m.η)⊡(Δt*E2⊡ϵ + m.η*old.ϵv)
    σ = E1⊡ϵ + m.η*(ϵv-old.ϵv)/Δt
    return σ, ϵv
end

function MMB.material_response(
        m::ViscoElastic, ϵ::SymmetricTensor{2},
        old::ViscoElasticState, Δt, args...)
    if Δt == zero(Δt)
        if norm(ϵ) ≈ 0
            return zero(ϵ), NaN*zero(get_stiffness(m.E1)), old
        else
            throw(NoLocalConvergence("Δt=0 is not allowed"))
        end
    end
    σ, ϵv = get_stress_ϵv(m,ϵ,old,Δt)
    dσdϵ = gradient(ϵ_->get_stress_ϵv(m,ϵ_,old,Δt)[1], ϵ)
    return σ, dσdϵ, ViscoElasticState(ϵv)
end

function MMB.differentiate_material!(diff::MMB.MaterialDerivatives, m::ViscoElastic, ϵ, old::ViscoElasticState, Δt, _, _, dσdϵ)
    tomandel!(diff.dσdϵ, dσdϵ)
    dσdⁿs = zeros(get_num_tensorcomponents(m), get_num_statevars(m))
    ForwardDiff.jacobian!(dσdⁿs, sv -> tovector(get_stress_ϵv(m, ϵ, fromvector(sv, old), Δt)[1]), tovector(old))
    ∂σ∂p = ForwardDiff.jacobian(p -> tovector(get_stress_ϵv(fromvector(p, m), ϵ, old, Δt)[1]), tovector(m))
    
    dsdⁿs = zeros(get_num_statevars(m), get_num_statevars(m))
    ForwardDiff.jacobian!(diff.dsdϵ, e -> tovector(ViscoElasticState(get_stress_ϵv(m, fromvector(e, ϵ), old, Δt)[2])), tovector(ϵ))
    ForwardDiff.jacobian!(dsdⁿs, sv -> tovector(ViscoElasticState(get_stress_ϵv(m, ϵ, fromvector(sv, old), Δt)[2])), tovector(old))
    ∂s∂p = ForwardDiff.jacobian(p -> tovector(ViscoElasticState(get_stress_ϵv(fromvector(p, m), ϵ, old, Δt)[2])), tovector(m))
    
    diff.dσdp .= ∂σ∂p .+ dσdⁿs * diff.dsdp
    diff.dsdp .= ∂s∂p .+ dsdⁿs * diff.dsdp
end