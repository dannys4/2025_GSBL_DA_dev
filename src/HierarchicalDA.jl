module HierarchicalDA

using BandedMatrices
using DocStringExtensions
using Distributions
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using OrdinaryDiffEq
using ProgressMeter
using QuadGK
using Random
using RecursiveArrayTools
using SparseArrays
using SpecialFunctions
using StaticArrays
using StatsBase
using Statistics
using TransportBasedInference2
using Trixi
using UnPack


import LinearMaps: LinearMap, FunctionMap
import UnPack: @unpack

include("tools/modulo_realnumbers.jl")
include("tools/unroll.jl")
include("tools/smooth_periodic.jl")
include("tools/vec2sol.jl")
include("tools/crps.jl")



include("distributions/generalized_gamma.jl")
include("distributions/extended_gamma.jl")


include("PA/PA.jl")
include("PA/jump_function.jl")

include("update_theta/flow_theta.jl")
include("update_theta/update_theta.jl")


include("update_x/vectors.jl")
include("update_x/obs_system.jl")
include("update_x/obs_constraint_system.jl")

include("update_x/covariance.jl")

include("update_x/enkf.jl")
include("update_x/localized_enkf.jl")

include("update_x/hierarchical_enkf.jl")
include("update_x/hierarchical_localized_enkf.jl")
include("update_x/update_x_hierarchical_enkf_separate.jl")
include("update_x/update_x_hierarchical_enkf_shared.jl")

# update utils
include("tools/update_utils.jl")

# Setup object for Trixi
include("trixi/trixi_system.jl")
include("trixi/generate_data.jl")
include("trixi/seqassim_trixi.jl")

# Linear advection
include("linear_advection/linear_advection.jl")

# Inviscid Burgers
include("inviscid_burgers/inviscid_burgers.jl")

# Shu-Osher equation
include("euler/euler.jl")
include("euler/seqassim_trixi_euler.jl")











end # module HierarchicalDA
