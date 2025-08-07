using HierarchicalDA
using HierarchicalDA: getĈX, ObsSystem
using LinearAlgebra
using Random
using LinearMaps
using SparseArrays
using TransportBasedInference2
using IterativeSolvers

##
rng = Xoshiro(283028)
Nx = 100
xgrid = [0;cumsum(rand(rng, Nx - 1))]
xgrid /= xgrid[end]
delta_y = 10
Lrad = 7
sigma_y = 0.2
delta_t = 0.05

##
yidx = 1:delta_y:Nx
Ny = length(yidx)
H = LinearMap(sparse(Matrix(1.0 * I, Nx, Nx)[1:delta_y:end, :]))
H_mat = Matrix(H)

##
@assert Int.(H * (1:size(H,2))) == yidx
@assert findall(!iszero, H' * (1:size(H,1))) == yidx

##
Ne = 50
ϵy = AdditiveInflation(Ny, sigma_y)
Cϵ = LinearMaps.UniformScalingMap(sigma_y, length(yidx))
Cϵ_mat = Matrix(Cϵ)
# This CX is replaced with the estimated state cov at each step
sys_y = ObsSystem(H, Cϵ)

Gxx(i, j) = 0.0
Loc = Localization(Nx, Lrad, Gxx, is_sparse=true)
enkf_mat = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t, delta_t, isiterative=false)
enkf_itr = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t, delta_t, isiterative=true)

##
Random.seed!(rng, 293208132)
ens = randn(rng, Nx, Ne)
Cx_true = cov(ens, dims=2)
Cx_est_mat = Matrix(getĈX(enkf_mat, ens, Nx, Ny))
@assert norm(Cx_est_mat - Cx_true)/norm(Cx_true) < 1e-14

##
# Generic case
workspace_sparsity = collect(1:Nx)
Cx_est_itr_op = getĈX(enkf_itr, ens, Nx, Ny; with_matrix=false, workspace_sparsity)
in_Cx_est_itr = 1.0*I(Nx)
Cx_est_itr = zeros(Nx, Nx)
for col_idx in axes(Cx_est_itr,2)
    mul!(@view(Cx_est_itr[:,col_idx]), Cx_est_itr_op, @view(in_Cx_est_itr[:,col_idx]))
end
@assert norm(Cx_est_itr - Cx_true)/norm(Cx_true) < 1e-14

## Kalman system matrix truth
sys_truth = H_mat * Cx_true * H_mat' + Cϵ_mat

##
enkf_mat.sys.CX = Cx_est_mat
sys_mat = Matrix(enkf_mat.sys)
@assert norm(sys_mat - sys_truth)/norm(sys_truth) < 1e-14
sys_fac = bunchkaufman!(sys_mat)

##
sys_op = enkf_itr.sys.H * Cx_est_itr_op * enkf_itr.sys.H' + enkf_itr.sys.Cϵ
@assert norm(Matrix(sys_op) - sys_truth)/norm(sys_truth) < 1e-14

##
# Sparse case
Gxx(i, j) = periodicmetric!(i, j, Nx) # Unlocalized
Loc = Localization(Nx, Lrad, Gxx, is_sparse=true)
enkf_mat = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t, delta_t, isiterative=false)
enkf_itr = LocEnKF(Ne, ϵy, sys_y, Loc, delta_t, delta_t, isiterative=true)

Random.seed!(rng, 293208132)
ens = Matrix(H' * randn(rng, Ny, Ne))
Cx_true = collect(cov(ens, dims=2) .* Loc.ρX)
Cx_est_mat = Matrix(getĈX(enkf_mat, ens, Nx, Ny))
@assert norm(Cx_est_mat - Cx_true)/norm(Cx_true) < 1e-14

##
workspace_sparsity = findall(isnan, enkf_itr.sys.H' * fill(NaN, size(enkf_itr.sys.H, 1)))
Cx_est_itr_op = getĈX(enkf_itr, ens, Nx, Ny; with_matrix=false, workspace_sparsity)
in_Cx_est_itr = 1.0*I(Nx)
Cx_est_itr = zeros(Nx, Nx)
for col_idx in axes(Cx_est_itr,2)
    mul!(@view(Cx_est_itr[:,col_idx]), Cx_est_itr_op, @view(in_Cx_est_itr[:,col_idx]))
end
@assert norm(Cx_est_itr - Cx_true)/norm(Cx_true) < 1e-14

## Kalman system matrix truth
sys_truth = H_mat * Cx_true * H_mat' + Cϵ_mat

##
enkf_mat.sys.CX = Cx_est_mat
sys_mat = Matrix(enkf_mat.sys)
@assert norm(sys_mat - sys_truth)/norm(sys_truth) < 1e-14
sys_fac = bunchkaufman!(sys_mat)

##
sys_op = enkf_itr.sys.H * Cx_est_itr_op * enkf_itr.sys.H' + enkf_itr.sys.Cϵ
@assert norm(Matrix(sys_op) - sys_truth)/norm(sys_truth) < 1e-14
id = collect(1.0*I(Ny))
out = similar(sys_truth)
for i in axes(out,2)
    cg!(@view(out[:,i]), sys_op, @view(id[:,i]), log=true, verbose=true, reltol=1e-11, abstol=1e-11, maxiter=50)
end
out