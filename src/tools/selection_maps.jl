using Base: size, getindex
using LinearMaps: _unsafe_mul!, MulStyle, FiveArg
export IdentityMap, SelectionMap, ZeroMap, diag_select

struct IdentityMap{T} <: LinearMap{T}
    side_len::Int
end
function IdentityMap(side_len::Int, _::Type{T}=Float64) where {T}
    IdentityMap{T}(side_len)
end
LinearMaps.issymmetric(::IdentityMap) = true
LinearMaps.MulStyle(::IdentityMap) = FiveArg()
Base.size(id::IdentityMap) = (id.side_len, id.side_len)
LinearAlgebra.adjoint(id::IdentityMap) = id
LinearAlgebra.transpose(id::IdentityMap) = id

function identity_mul!(y, ::IdentityMap, x)
    @inbounds @simd for idx in eachindex(x, y)
        y[idx] = x[idx]
    end
end

function identity_mul!(y, ::IdentityMap, x, alpha, beta)
    @inbounds @simd for idx in eachindex(x, y)
        y[idx] = muladd(beta, y[idx], alpha * x[idx])
    end
    y
end

LinearMaps._unsafe_mul!(y::AbstractMatrix, H::IdentityMap, x::AbstractMatrix) = identity_mul!(y, H, x)
LinearMaps._unsafe_mul!(y::AbstractVector, H::IdentityMap, x::AbstractVector) = identity_mul!(y, H, x)

LinearMaps._unsafe_mul!(y::AbstractMatrix, H::IdentityMap, x::AbstractMatrix, alpha, beta) = identity_mul!(y, H, x, alpha, beta)
LinearMaps._unsafe_mul!(y::AbstractVector, H::IdentityMap, x::AbstractVector, alpha, beta) = identity_mul!(y, H, x, alpha, beta)

struct ZeroMap{T} <: LinearMap{T}
    size::Tuple{Int,Int}
end
function ZeroMap(rows::Int, cols::Int, _::Type{T}=Float64) where {T}
    ZeroMap{T}((rows, cols))
end
LinearMaps.issymmetric(z::ZeroMap) = z.size[1] == z.size[2]
LinearMaps.MulStyle(::ZeroMap) = FiveArg()
Base.size(z::ZeroMap) = z.size
function Base.eltype(::ZeroMap{T}) where {T}
    T
end
LinearAlgebra.adjoint(z::ZeroMap{T}) where {T} = ZeroMap{T}((z.size[2], z.size[1]))
LinearAlgebra.transpose(z::ZeroMap) = adjoint(z)

function zero_mul!(y, beta)
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    else
        lmul!(beta, y)
    end
    y
end
# LinearMaps._unsafe_mul!(y, ::ZeroMap, _) = zero_mul!(y, false)
LinearMaps._unsafe_mul!(y::AbstractVector{T}, ::ZeroMap{T}, x::AbstractVector{T}, alpha=true, beta=false) where {T} = zero_mul!(y, beta)
LinearMaps._unsafe_mul!(y::AbstractMatrix{T}, ::ZeroMap{T}, x::AbstractMatrix{T}, alpha=true, beta=false) where {T} = zero_mul!(y, beta)
# LinearMaps._unsafe_mul!(y::MV, ::ZeroMap{T}, ::MV, alpha=true, beta=false) where {T,MV<:AbstractMatrix{T}} = zero_mul!(y, beta)



# If IsOut true, y = H*x <=> y == x[idxs]. If false, y = H*x <=> y[idxs] == x
struct SelectionMap{IsOut,T,V<:AbstractVector{Int}} <: LinearMap{T}
    idxs::V
    size::Tuple{Int,Int}
end

LinearMaps.MulStyle(::SelectionMap) = FiveArg()

function SelectionMap(idxs::_V, selection::Symbol; in_size=maximum(idxs), _::Type{_T}=Float64) where {_V,_T}
    @assert selection in (:in, :out) "Can only select in or out"
    @assert maximum(idxs) <= in_size && all(>(0), idxs)
    sort_idxs = sort(idxs)
    IsOut = selection == :out
    SelectionMap{IsOut,_T,_V}(sort_idxs, (length(sort_idxs), in_size))
end

Base.size(s::SelectionMap) = s.size

function Base.getindex(A::SelectionMap{true}, row::Int, ::Colon)
    row > size(A, 1) && BoundsError()
    row <= length(A.idxs) ? sparsevec([A.idxs[row]], [true], A.size[2]) : spzeros(A.size)
end

function Base.getindex(A::SelectionMap{false}, ::Colon, col::Int)
    Base.getindex(A', col, :)
end

function Base.Matrix(A::SelectionMap{true})
    sparse(eachindex(A.idxs), A.idxs, ones(Bool, length(A.idxs)), A.size...)
end

function Base.Matrix(A::SelectionMap{false})
    Matrix(A')'
end

function Base.getindex(A::T, I1::V1, I2::V2) where {T<:Union{IdentityMap,LinearMaps.UniformScalingMap},V1,V2}
    if (A isa LinearMaps.UniformScalingMap && !isone(A.λ)) || !(V1 == Colon || V2 == Colon)
        return getindex(Matrix(A), I1, I2)
    end
    IsOut = V2 == Colon
    if IsOut
        return SelectionMap{IsOut,Float64,V1}(deepcopy(I1), (length(I1), A.M))
    else
        return SelectionMap{IsOut,Float64,V2}(deepcopy(I2), (A.M, length(I2)))
    end
end

function select_mul!(y::AbstractVector, H::SelectionMap{IsOut}, x::AbstractVector) where {IsOut}
    if IsOut
        for (y_idx, x_idx) in enumerate(H.idxs)
            y[y_idx] = x[x_idx]
        end
    else
        for (x_idx, y_idx) in enumerate(H.idxs)
            y[y_idx] = x[x_idx]
        end
    end
    return y
end

function select_mul!(y, H::SelectionMap{IsOut}, x, alpha, beta) where {IsOut}
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    else
        rmul!(y, beta)
    end
    if IsOut
        for (y_idx, x_idx) in enumerate(H.idxs)
            y[y_idx] = muladd(alpha, x[x_idx], y[y_idx])
        end
    else
        for (x_idx, y_idx) in enumerate(H.idxs)
            y[y_idx] = muladd(alpha, x[x_idx], y[y_idx])
        end
    end
    return y
end

isout_container(x, T, axis) = similar(x, T, axis)
isout_container(::SparseVector{T}, ::Type{T}, axis) where {T} = Vector{T}(undef, axis.stop)

function Base.:(*)(A::SelectionMap{IsOut}, x::AbstractVector) where {IsOut}
    LinearMaps.check_dim_mul(A, x)
    T = promote_type(eltype(A), eltype(x))
    if IsOut
        y = isout_container(x, T, axes(A)[1])
    else
        y = sparsevec(A.idxs, ones(T, length(A.idxs)), A.size[1])
    end
    return @inbounds mul!(y, A, x)
end

LinearMaps._unsafe_mul!(y::AbstractMatrix, H::SelectionMap, x::AbstractMatrix) = select_mul!(y, H, x, true, false)
LinearMaps._unsafe_mul!(y::AbstractVector, H::SelectionMap, x::AbstractVector) = select_mul!(y, H, x)

LinearMaps._unsafe_mul!(y::AbstractMatrix, H::SelectionMap, x::AbstractMatrix, alpha, beta) = select_mul!(y, H, x, alpha, beta)
LinearMaps._unsafe_mul!(y::AbstractVector, H::SelectionMap, x::AbstractVector, alpha, beta) = select_mul!(y, H, x, alpha, beta)


function LinearAlgebra.adjoint(H::SelectionMap{IsOut,T,V}) where {IsOut,T,V}
    sz = H.size
    SelectionMap{!IsOut,T,V}(deepcopy(H.idxs), (sz[2], sz[1]))
end

LinearAlgebra.transpose(H::SelectionMap) = adjoint(H)

function diag_select(diag_idx, num_cols, _::Type{T}=Float64) where {T}
    id_size = num_cols - abs(diag_idx)
    id = IdentityMap(id_size, T)
    corner_sz = abs(diag_idx)
    corner_pad = ZeroMap(corner_sz, corner_sz, T)
    col_pad = ZeroMap(num_cols - corner_sz, corner_sz, T)
    row_pad = ZeroMap(corner_sz, num_cols - corner_sz, T)
    if diag_idx > 0
        return [col_pad id; corner_pad row_pad]
    elseif diag_idx < 0
        return [row_pad corner_pad; id col_pad]
    else
        return id
    end
end