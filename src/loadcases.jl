function symmetric_components(ij::NTuple{2, Int})
    return ij[1] < ij[2] ? reverse(ij) : ij
end

construct_tensor(ϵ::AbstractTensor, _) = ϵ

function construct_tensor(ϵ::Number, ij)
    i, j = symmetric_components(ij)
    return SymmetricTensor{2,3}((k, l) -> k == i && l == j ? ϵ : zero(ϵ))
end

"""
    runstrain(m, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate fully strain-controlled loading from zero to `ϵ_end` strain, and return σ[i,j]. 
If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 

Returns the vector of stresses as well as the final state. 
"""
function runstrain(m, ϵ_ij_end::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    ϵ_end = construct_tensor(ϵ_ij_end, ij)
    i, j = symmetric_components(ij)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵ_end * scale
        σ, _, state = material_response(m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

"""
    runstrain_diff(m, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate fully strain-controlled loading from zero to `ϵ_end` strain, and return σ[i,j] and its derivatives
wrt. to the material parameters. If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 

Returns the vector of stresses, the final state, the derivatives `dσᵢⱼ/dp` for all time steps, and the final 
derivatives `diff::MaterialDerivatives`. 
"""
function runstrain_diff(m, ϵ_ij_end::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    ϵ_end = construct_tensor(ϵ_ij_end, ij)
    i, j = symmetric_components(ij)
    mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
    mscale = i == j ? 1.0 : 1/√2
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = MaterialDerivatives(m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_num_params(m))
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

Returns the vector of stresses and the final state variables.
"""
function runstresstate(stress_state, m, ϵend::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    ϵt = construct_tensor(ϵend, ij)
    i, j = symmetric_components(ij)
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        σ, _, state = material_response(stress_state, m, ϵ, state, Δt, cache)
        σv[k + 1] = σ[i, j]
    end
    return σv, state
end

"""
    runstresstate_diff(stress_state, m, ϵ_end::Union{Number, AbstractTensor}, ij::NTuple{2, Int}, t_end, num_steps)

Simulate the `stress_state` from zero to `ϵ_end` strain, and return σ[i,j] and its derivatives
wrt. to the material parameters. If `ϵ_end::Number` is passed, loading is applied to the strain `ϵ_end eᵢ⊗eⱼ`. 

Returns the vector of stresses, the final state, the derivatives `dσᵢⱼ/dp` for all time steps, and the final 
derivatives `diff::StressStateDerivatives`.
"""
function runstresstate_diff(stress_state, m, ϵend::Union{Number, AbstractTensor}, ij, t_end, num_steps)
    ϵt = construct_tensor(ϵend, ij)
    i, j = symmetric_components(ij)
    mind = Tensors.DEFAULT_VOIGT_ORDER[3][j, i]
    mscale = i == j ? 1.0 : 1/√2
    state = initial_material_state(m)
    cache = allocate_material_cache(m)
    diff = StressStateDerivatives(stress_state, m)
    extras = allocate_differentiation_output(m)
    Δt = t_end / num_steps
    σv = zeros(num_steps + 1)
    dσdp = zeros(num_steps + 1, get_num_params(m))
    for (k, scale) in enumerate(range(0, 1, num_steps + 1)[2:end])
        ϵ = ϵt * scale
        old_state = state
        σ, _, state = differentiate_material!(diff, stress_state, m, ϵ, old_state, Δt, cache, extras)
        σv[k + 1] = σ[i, j]
        dσdp[k + 1, :] .= diff.dσdp[mind, :] .* mscale
    end
    return σv, state, dσdp, diff
end
