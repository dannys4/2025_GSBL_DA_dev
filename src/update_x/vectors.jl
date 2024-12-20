export ObsConstraintVector, observation, constraint

const ObsConstraintVector = ArrayPartition

# ObsConstraintVector(y, s) = ArrayPartition(y, s)

ObsConstraintVector(Ny::Int64, Ns::Int64) = ObsConstraintVector(zeros(Ny), zeros(Ns))

observation(u::ObsConstraintVector) = u.x[1]
constraint(u::ObsConstraintVector) = u.x[2]
