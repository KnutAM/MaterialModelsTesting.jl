using MaterialModelsTesting
using Test
using Tensors, FiniteDiff
using MaterialModelsBase
using MaterialModelsTesting:
    LinearElastic, ViscoElastic, test_derivative, obtain_numerical_material_derivative!,
    runstrain, runstrain_diff, runstresstate, runstresstate_diff

@testset "MaterialModelsTesting.jl" begin
    elastic = LinearElastic(0.52, 0.77)
    viscoelastic = ViscoElastic(elastic, LinearElastic(0.33, 0.54), 0.36)
    for m in (elastic, viscoelastic)
        @testset "testsuite" begin
            MaterialModelsTesting.test_material(m)
        end 
        @testset "Initial response" begin
            ϵ = rand(SymmetricTensor{2,3}) * 1e-3
            Δt = 1e-2
            test_derivative(m, ϵ, initial_material_state(m), Δt; 
                numdiffsettings = (fdtype = Val{:central},),
                comparesettings = ()
                )
            diff = MaterialDerivatives(m)
            copy!(diff.dsdp, rand(size(diff.dsdp)...))
            test_derivative(m, ϵ, initial_material_state(m), Δt; 
                numdiffsettings = (fdtype = Val{:central},),
                comparesettings = (),
                diff)
        end
        @testset "After shear loading" begin
            ϵ21 = 0.01; num_steps = 10; t_end = 0.01
            stressfun(p) = runstrain(fromvector(p, m), ϵ21, (2, 1), t_end, num_steps)[1]
            dσ21_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-6)
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
        for (stress_state, ij) in (
                (UniaxialStress(), (1,1)), (UniaxialStrain(), (1,1)), 
                (UniaxialNormalStress(), (1,1)), (UniaxialNormalStress(), (2,1)),
                (PlaneStress(), (2, 2)), (PlaneStrain(), (2, 1)),
                (GeneralStressState(SymmetricTensor{2,3,Bool}((true, false, false, false, true, true)), rand(SymmetricTensor{2,3})), (2,2))
                )
            @testset "$(nameof(typeof(stress_state))), (i,j) = ($(ij[1]), $(ij[2]))" begin
                ϵij = 0.01; num_steps = 1; t_end = 0.01
                stressfun(p) = runstresstate(stress_state, fromvector(p, m), ϵij, ij, t_end, num_steps)[1]
                dσij_dp_num = FiniteDiff.finite_difference_jacobian(stressfun, tovector(m), Val{:central}; relstep = 1e-6)
                σv, state, dσij_dp, diff = runstresstate_diff(stress_state, m, ϵij, ij, t_end, num_steps)
                @test dσij_dp ≈ dσij_dp_num
            end
        end
    end
end
