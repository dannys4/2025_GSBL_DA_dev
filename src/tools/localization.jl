using TransportBasedInference2: Localization

function TransportBasedInference2.Localization(Nxvar::Int, Lrad::Int, Gxx::Function, Nvar::Int; use_linearmap=false, loc_kwargs...)
    loc = Localization(Nxvar, Lrad, Gxx; loc_kwargs...)
    if use_linearmap
        L = LinearMap(loc.ρX)
        Localization(kron(L, ones(Bool, Nvar, Nvar)))
    else
        Localization(kron(loc.ρX, ones(Bool, Nvar, Nvar)))
    end
end