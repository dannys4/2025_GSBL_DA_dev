getĈX(enkf::HEnKF, X, Ny, Nx) = EmpiricalCov(@view(X[Ny+1:Ny+Nx, :]); with_matrix = true)
getĈX(enkf::HLocEnKF, X, Ny, Nx) = LocalizedEmpiricalCov(@view(X[Ny+1:Ny+Nx, :]), enkf.Loc; with_matrix = true)
