export unroll

function unroll(idx, L, K)
    @assert prod(x -> x <= L, idx) == true
    unroll_idx = Int64[]
    for k = 1:K
        unroll_idx = vcat(unroll_idx, (k - 1) * L .+ collect(idx))
    end
    return unroll_idx
end
