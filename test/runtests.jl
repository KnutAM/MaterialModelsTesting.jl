using MaterialModelsTesting
using Test
using Tensors, FiniteDiff
using MaterialModelsBase
using MaterialModelsTesting:
    LinearElastic, ViscoElastic, NeoHooke, 
    test_derivative, obtain_numerical_material_derivative!,
    runstrain, runstrain_diff, runstresstate, runstresstate_diff,
    construct_tensor

@testset "MaterialModelsTesting.jl" begin
    elastic = LinearElastic(0.52, 0.77)
    viscoelastic = ViscoElastic(elastic, LinearElastic(0.33, 0.54), 0.36)
    hyperelastic = NeoHooke(; G = 0.55, K = 0.76)
    @testset "$(typeof(m))" for m in (elastic, hyperelastic, viscoelastic)
        TB = MaterialModelsBase.get_tensorbase(m)
        @testset "testsuite" begin
            MaterialModelsTesting.test_material(m)
        end
        @testset "Initial response" begin
            #ϵ = 
            strain = if TB == SymmetricTensor{2,3}
                rand(SymmetricTensor{2,3}) * 1e-3
            else
                one(Tensor{2,3}) + rand(Tensor{2,3}) * 1e-3
            end
            Δt = 1e-2
            test_derivative(m, strain, initial_material_state(m), Δt; 
                numdiffsettings = (fdtype = Val{:central},),
                comparesettings = ()
                )
            diff = MaterialDerivatives(m)
            copy!(diff.dsdp, rand(size(diff.dsdp)...))
            test_derivative(m, strain, initial_material_state(m), Δt; 
                numdiffsettings = (fdtype = Val{:central},),
                comparesettings = (),
                diff)
        end
        @testset "After shear loading" begin
            ϵ21 = 0.01; num_steps = 10; t_end = 0.01
            stressfun(p) = runstrain(fromvector(p, m), ϵ21, (2, 1), t_end, num_steps)[1]
            dσ21_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-8)
            σv, state, dσ21_dp, diff = runstrain_diff(m, ϵ21, (2, 1), t_end, num_steps)
            @test σv ≈ stressfun(tovector(m))
            @test dσ21_dp ≈ dσ21_dp_num
        end
        @testset "FullStressState" begin
            ϵ21 = 0.01; num_steps = 10; t_end = 0.01
            ss = FullStressState()
            # Check that we get the same result for runstresstate and runstrain
            
            σ_ss, s_ss = runstresstate(ss, m, ϵ21, (2, 1), t_end, num_steps)
            σ, s = runstrain(m, ϵ21, (2, 1), t_end, num_steps)
            @test σ_ss ≈ σ
            @test tovector(s_ss) ≈ tovector(s)
        end
        
        ϵij = 0.01
        for (stress_state, ij) in (
                (UniaxialStress(), (1,1)), (UniaxialStrain(), (1,1)), 
                (UniaxialNormalStress(), (1,1)), (UniaxialNormalStress(), (2,1)),
                (PlaneStress(), (2, 2)), (PlaneStrain(), (2, 1)),
                (GeneralStressState(TB(SymmetricTensor{2,3,Bool}((true, false, false, false, true, true))), ϵij * rand(TB); max_iter = 100), (2,2))
                )
            @testset "$(nameof(typeof(stress_state))), (i,j) = ($(ij[1]), $(ij[2]))" begin
                num_steps = 1; t_end = 0.01
                stressfun(p) = runstresstate(stress_state, fromvector(p, m), ϵij, ij, t_end, num_steps)[1]
                dσij_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-6)
                σv, state, dσij_dp, diff = runstresstate_diff(stress_state, m, ϵij, ij, t_end, num_steps)
                @test isapprox(dσij_dp, dσij_dp_num; rtol = 1e-6)
            end
        end
    end
end
