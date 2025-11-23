function symmetric_components(::Type{<:SymmetricTensor}, ij::NTuple{2, Int})
    return ij[1] < ij[2] ? reverse(ij) : ij
end
symmetric_components(::Type{<:Tensor}, ij::NTuple{2, Int}) = ij

function construct_tensor(::Type{TB}, ϵ::AbstractTensor, _) where {TB}
    if !isa(ϵ, TB)
        error("Incompatible types passed to construct tensor")
    elseif isa(ϵ, Tensor)
        return ϵ + one(ϵ)
    else
        return ϵ
    end
end

function construct_tensor(::Type{TB}, ϵ::Number, ij) where {TB <: Union{Tensor{2,3}, SymmetricTensor{2,3}}}
    i, j = symmetric_components(TB, ij)
    return construct_tensor(TB, TB((k, l) -> k == i && l == j ? ϵ : zero(ϵ)), nothing)
end

"""
    runstrain(m, TensorBase, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate fully strain-controlled loading from zero to `ϵ_end` strain, and return σ[i,j]. 
If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 
`TensorBase` should be `SymmetricTensor{2,3}` for small strain and `Tensor{2,3}` for finite strain. For finite strain,
the deformation gradient `I2 + ϵ_end eᵢ⊗eⱼ` is applied. 

Returns the vector of stresses as well as the final state. 
"""
function runstrain(m, ϵ_ij_end::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    TB = MMB.get_tensorbase(m)
    ϵ_end = construct_tensor(TB, ϵ_ij_end, ij)
    i, j = symmetric_components(TB, ij)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(get_vector_eltype(m), num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵ_end * scale
        σ, _, state = material_response(m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

"""
    runstrain_diff(m, TensorBase, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate fully strain-controlled loading from zero to `ϵ_end` strain, and return σ[i,j] and its derivatives
wrt. to the material parameters. If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 
`TensorBase` should be `SymmetricTensor{2,3}` for small strain and `Tensor{2,3}` for finite strain. For finite strain,
the deformation gradient `I2 + ϵ_end eᵢ⊗eⱼ` is applied. 

Returns the vector of stresses, the final state, the derivatives `dσᵢⱼ/dp` for all time steps, and the final 
derivatives `diff::MaterialDerivatives`. 
"""
function runstrain_diff(m, ϵ_ij_end::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    TB = MMB.get_tensorbase(m)
    ϵ_end = construct_tensor(TB, ϵ_ij_end, ij)
    i, j = symmetric_components(TB, ij)
    if TB == SymmetricTensor{2,3}    
        mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
        mscale = i == j ? 1.0 : 1/√2
    else
        mind = Tensors.DEFAULT_VOIGT_ORDER[3][i, j]
        mscale = 1.0
    end
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = MaterialDerivatives(m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_vector_length(m))
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵ_end * scale
        old_state = state
        σ, dσdϵ, state = material_response(m, ϵ, old_state, Δt, cache, extras)
        differentiate_material!(diff, m, ϵ, old_state, Δt, cache, extras, dσdϵ)
        σv[k + 1] = σ[i, j]
        dσdp[k + 1, :] .= diff.dσdp[mind, :] .* mscale
    end
    return σv, state, dσdp, diff
end

"""
    runstresstate(stress_state, m, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate the `stress_state` from zero to `ϵ_end` strain, and return σ[i,j].
If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 
For finite strain, the deformation gradient `I2 + ϵ_end eᵢ⊗eⱼ` is applied. 

Returns the vector of stresses and the final state variables.
"""
function runstresstate(stress_state, m, ϵend::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    TB = MMB.get_tensorbase(m)
    ϵt = construct_tensor(TB, ϵend, ij)
    i, j = symmetric_components(TB, ij)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(get_vector_eltype(m), num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        σ, _, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

"""
    runstresstate_diff(stress_state, m, TensorBase, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate the `stress_state` from zero to `ϵ_end` strain, and return σ[i,j] and its derivatives
wrt. to the material parameters. If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`.

Returns the vector of stresses, the final state, the derivatives `dσᵢⱼ/dp` for all time steps, and the final 
derivatives `diff::StressStateDerivatives`.
"""
function runstresstate_diff(stress_state, m, ϵend::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    TB = MMB.get_tensorbase(m)
    ϵt = construct_tensor(TB, ϵend, ij)
    i, j = symmetric_components(TB, ij)
    if TB == SymmetricTensor{2,3}    
        mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
        mscale = i == j ? 1.0 : 1/√2
    else
        mind = Tensors.DEFAULT_VOIGT_ORDER[3][i, j]
        mscale = 1.0
    end
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = StressStateDerivatives(stress_state, m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_vector_length(m))
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        old_state = state
        σ, _, state = differentiate_material!(diff, stress_state, m, ϵ, old_state, Δt, cache, extras)
        σv[k + 1] = σ[i, j]
        dσdp[k + 1, :] .= diff.dσdp[mind, :] .* mscale
    end
    return σv, state, dσdp, diff
end
