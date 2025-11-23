"""
    test_derivative(m, ϵ, state, Δt; comparesettings = (), numdiffsettings = (), diff = MaterialDerivatives(m))

This function is used to compare the analytically implemented derivative (calling `differentiate_material!(diff, m, ϵ, args...)`)
with the result from `obtain_numerical_material_derivative!(diff, m, ϵ, args...; numdiffsettings...)`.

* `diff::MaterialDerivatives` can be passed to check that the old value of `diff.dsdp` is correctly accounted for.
* `comparesettings::NamedTuple` are passed as `kwargs` to [`compare_derivatives`](@ref), which compares the two matrices. Please see its docstring for further details.
* `numdiffsettings::NamedTuple` are passed as `kwargs` to [`obtain_numerical_material_derivative!`](@ref), please see its docstring for further details. 
"""
function test_derivative(m, ϵ, state, Δt; comparesettings = (), numdiffsettings = (), diff = MaterialDerivatives(m))
    diff_num = MaterialDerivatives(m)
    copy!(diff_num.dsdp, diff.dsdp)

    cache = allocate_material_cache(m)
    extras = allocate_differentiation_output(m)
    _, dσdϵ, _ = material_response(m, ϵ, state, Δt, cache, extras)
    differentiate_material!(diff, m, ϵ, state, Δt, cache, extras, dσdϵ)
    obtain_numerical_material_derivative!(diff_num, m, ϵ, state, Δt; numdiffsettings...)
    y, Δx = material_derivative_scaling(diff, m, ϵ, state, Δt; numdiffsettings...)
    for key in fieldnames(typeof(diff))
        @testset "$key" begin
            scaled_error, maxtol = compare_derivatives(getfield(diff, key), getfield(diff_num, key), y[key], Δx[key]; comparesettings...)
            check = all(x -> x ≤ 1, scaled_error)
            @test check
            if !check
                println(key, " failed (maxtol = ", maxtol, ")")
                println("Printing derivative, numerical derivative, relative diff, and scaled error")
                display(getfield(diff, key))
                display(getfield(diff_num, key))
                display((getfield(diff, key) .- getfield(diff_num, key)) ./ (1e-100 .+ abs.(getfield(diff, key)) + abs.(getfield(diff_num, key))))
                display(scaled_error)
                println("Done printing for failed test above")
            end
        end
    end
end

"""
    obtain_numerical_material_derivative!(deriv, m, ϵ, old, Δt; fdtype = Val{:forward}, kwargs...)

Obtain the numerical derivative of the material `m` at the strain `ϵ` and old state variables, `old`,
for a time step `Δt`. `fdtype` and `kwargs...` are passed to `FiniteDiff.finite_difference_jacobian`. 
"""
function obtain_numerical_material_derivative!(
        deriv::MMB.MaterialDerivatives,
        m::AbstractMaterial, ϵ, old, Δt; 
        fdtype = Val{:forward}, 
        typeconvert = identity, 
        kwargs...
        )
    cache = allocate_material_cache(m)
    p = map(typeconvert, tovector(m))
    ⁿs = map(typeconvert, tovector(old))
    e = map(typeconvert, tovector(ϵ))
    
    function numjac(f::F, x) where {F}
        sz = (length(f(x)), length(x))
        prod(sz) == 0 && return zeros(sz...)
        return FiniteDiff.finite_difference_jacobian(f, x, fdtype; kwargs...)
    end
    numjac!(J, f::F, x) where {F} = copy!(J, numjac(f, x))

    funs = NamedTuple{(:σ, :s)}(
        (s = svec -> tovector(material_response(m, ϵ, fromvector(svec, old), Δt, cache)[i]), # f(s)
         p = pvec -> tovector(material_response(fromvector(pvec, m), ϵ, old, Δt, cache)[i]), # f(p)
         ϵ = evec -> tovector(material_response(m, fromvector(evec, ϵ), old, Δt, cache)[i])) # f(ϵ)
         for i in (1, 3)
    )

    dⁿsdp = copy(deriv.dsdp) # Copy not needed as dsdp field updated last, but safer in this test routine. 

    # Fill all partial derivatives
    numjac!(deriv.dσdϵ, funs.σ.ϵ, e)
    dσdⁿs = numjac(funs.σ.s, ⁿs)
    #numjac!(deriv. dσdⁿs, funs.σ.s, ⁿs) # not needed in deriv
    numjac!(deriv.dσdp, funs.σ.p, p)

    numjac!(deriv.dsdϵ, funs.s.ϵ, e)
    #numjac!(deriv. dsdⁿs, funs.s.s, ⁿs) # not needed in deriv
    dsdⁿs = numjac(funs.s.s, ⁿs)
    numjac!(deriv.dsdp, funs.s.p, p)

    # Adjust for dependence on previous state variable, i.e. ⁿs = ⁿs(p)
    deriv.dσdp .+= dσdⁿs * dⁿsdp
    deriv.dsdp .+= dsdⁿs * dⁿsdp

end

function material_derivative_scaling(::MMB.MaterialDerivatives, m, ϵ, old, Δt; fdtype = Val{:forward}, relstep = FiniteDiff.default_relstep(fdtype, eltype(ϵ)), absstep = relstep, kwargs...)
    Δp = max.(relstep * tovector(m), absstep)
    Δe = max.(relstep * tovector(ϵ), absstep)
    Δx = Dict(
        :dσdp => Δp, :dsdp => Δp,
        :dσdϵ => Δe, :dsdϵ => Δe)
    σ, _, state = material_response(m, ϵ, old, Δt)
    y = Dict(
        :dσdp => tovector(σ), :dsdp => tovector(state),
        :dσdϵ => tovector(σ), :dsdϵ => tovector(state))
    return y, Δx
end

"""
    obtain_numerical_material_derivative!(ssd, stress_state, m, ϵ, old, Δt; fdtype = Val{:forward}, kwargs...)

Obtain the numerical derivative of the material `m` considering the `stress_state` iterations at the strain 
`ϵ` and old state variables, `old`, for a time step `Δt`. `fdtype` and `kwargs...` are passed to `FiniteDiff.finite_difference_jacobian`. 
"""
function obtain_numerical_material_derivative!(
        ssd::MMB.StressStateDerivatives, 
        stress_state::AbstractStressState, 
        m::AbstractMaterial, ϵ, old, Δt; 
        fdtype = Val{:forward}, kwargs...
        )
    cache = allocate_material_cache(m)
    p = tovector(m)
    ⁿs = tovector(old)

    function numjac(f::F, x) where {F}
        sz = (length(f(x)), length(x))
        prod(sz) == 0 && return zeros(sz...)
        return FiniteDiff.finite_difference_jacobian(f, x, fdtype; kwargs...)
    end
    numjac!(J, f::F, x) where {F} = copy!(J, numjac(f, x))
    
    _tovector(t::AbstractTensor{<:Any, 2}) = tovector(MMB.expand_tensordim(stress_state, t))
    _tovector(t::AbstractTensor{<:Any, 1}) = tovector(MMB.expand_tensordim(stress_state, t))
    _tovector(t::Any) = tovector(t)
    funs = NamedTuple{(:σ, :ϵ, :s)}(
        (s = svec -> _tovector(material_response(stress_state, m, ϵ, fromvector(svec, old), Δt, cache)[i]), # f(s)
         p = pvec -> _tovector(material_response(stress_state, fromvector(pvec, m), ϵ, old, Δt, cache)[i])) # f(p)
         for i in (1, 4, 3)
    )
    
    dⁿsdp = copy(ssd.mderiv.dsdp)

    #numjac!(ssd.mderiv. dσdⁿs, funs.σ.s, ⁿs)
    dσdⁿs = numjac(funs.σ.s, ⁿs)
    numjac!(ssd.dσdp, funs.σ.p, p); ssd.dσdp .+= dσdⁿs * dⁿsdp

    dϵdⁿs = numjac(funs.ϵ.s, ⁿs);
    numjac!(ssd.dϵdp, funs.ϵ.p, p); ssd.dϵdp .+= dϵdⁿs * dⁿsdp
    
    dsdⁿs = numjac(funs.s.s, ⁿs)
    numjac!(ssd.mderiv.dsdp, funs.s.p, p); ssd.mderiv.dsdp .+= dsdⁿs * dⁿsdp
end

"""
    compare_derivatives(dydx::Matrix, dydx_num::Matrix, y::Vector, Δx::Vector; tolscale = 1)

Comparing derivatives with numerical values can be tricky, due to catastrophic cancellation easily 
making the numerical derivatives inaccurate. This function uses the following formula to determine
the appropriate tolerance for comparing the derivatives, which is based on the values of the 
differentiated function, 'y', the pertubation for each entry, `Δx`. 'atol = 2 * eps(y) / Δx`.

It returns the error scaled by the tolerance for all entries, `e`, such that the derivatives
can be considered equal if `all(x -> x ≤ 1, e)`
"""
function compare_derivatives(dydx::AbstractMatrix, dydx_num::AbstractMatrix, y::Vector, Δx::Vector; 
        tolscale = 1, 
        atol_min = max(eps(maximum(abs, dydx_num)), sqrt(eps(zero(eltype(dydx_num))))),
        rtol_min = cbrt(eps(eltype(y))),
        print_tol = false,
        )
    tolmatrix = similar(dydx)
    scaled_error = similar(dydx)
    maxtol = zero(eltype(y))
    for (i, (dyi_dx, dyi_dx_num, yi)) in enumerate(zip(eachrow(dydx), eachrow(dydx_num), y))
        for (j, (dyi_dxj, dyi_dxj_num, Δxj)) in enumerate(zip(dyi_dx, dyi_dx_num, Δx))
            atol = max(tolscale * 2 * eps(yi) / Δxj, atol_min, rtol_min * abs(dyi_dxj))
            scaled_error[i, j] = abs(dyi_dxj - dyi_dxj_num) / atol
            maxtol = max(maxtol, atol)
            tolmatrix[i, j] = atol
        end
    end
    if print_tol
        display(tolmatrix)
    end
    return scaled_error, maxtol
end

"""
    are_derivatives_equal(args...; maxtol = Inf, kwargs...)

Calls [`compare_derivatives`](@ref) (forwarding `args` and `kwargs`) and checks if the 
derivatives are equal.
The maximum allowed absolute tolerance can also be checked by setting `maxtol`.
"""
function are_derivatives_equal(args...; maxtol = nothing, kwargs...)
    scaled_error, highest_tol = compare_derivatives(args...; kwargs...)
    all_approx_equal = all(x -> x ≤ 1, scaled_error)
    if !all_approx_equal
        @show argmax(scaled_error), maximum(scaled_error)
    end
    @test all_approx_equal
    if maxtol !== nothing
        @test highest_tol ≤ maxtol
    end
end
