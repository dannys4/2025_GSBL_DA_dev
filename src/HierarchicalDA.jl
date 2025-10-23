module HierarchicalDA

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
using JLD2
using Dates


import LinearMaps: LinearMap, FunctionMap
import UnPack: @unpack

using TransportBasedInference2: SeqFilter

abstract type HierarchicalSeqFilter <: TransportBasedInference2.SeqFilter end
abstract type AbstractEmpiricalCov <: LinearMaps.LinearMap{Float64} end
LinearMaps.issymmetric(::AbstractEmpiricalCov) = true
LinearMaps.ishermitian(::AbstractEmpiricalCov) = true
LinearMaps.MulStyle(::AbstractEmpiricalCov) = LinearMaps.FiveArg()
Base.size(C::AbstractEmpiricalCov) = (C.Nx, C.Nx)

struct TrixiSystem{Eqns<:Trixi.AbstractEquations,Solver,MeshT,Semi<:Trixi.AbstractSemidiscretization}
    equations::Eqns
    dg::Solver
    mesh::MeshT
    semi::Semi
end
export TrixiSystem

include("tools/tools.jl")

include("distributions/generalized_gamma.jl")
include("distributions/extended_gamma.jl")

include("covariance/empirical.jl")
include("covariance/localized_empirical.jl")

include("observation_system/vectors.jl")
include("observation_system/obs_system.jl")
include("observation_system/obs_constraint_system.jl")

include("update_theta/flow_theta.jl")
include("update_theta/update_theta.jl")

include("filter/filter.jl")

include("update_x/hierarchical_separate.jl")
include("update_x/hierarchical_shared.jl")

include("trixi/tools.jl")
include("trixi/generate_data.jl")
include("trixi/seqassim_trixi.jl")


# Add equation systems
include("equations/linear_advection.jl") # Linear advection
include("equations/inviscid_burgers.jl") # Inviscid Burgers
include("equations/euler.jl") # Shu-Osher equation
include("equations/kpp.jl") # KPP equation

end # module HierarchicalDA
