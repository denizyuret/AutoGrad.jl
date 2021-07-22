# params(f) Based on deepcopy_internal:

params(f) = (ps=Param[]; params_internal(f,ps,IdDict()); ps)

# Tapes can only have params at the top level, so here is a more efficient implementation for tapes:
params(t::Tape) = (n.Value for n in t.list if n.Value isa Param)

params_internal(p::Param, ps::Vector{Param}, d::IdDict) = if !haskey(d,p); d[p]=true; push!(ps,p); end

params_internal(x::Union{Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,UnionAll,Task,Regex},
                ps::Vector{Param}, stackdict::IdDict) = return
params_internal(x::Tuple, ps::Vector{Param}, stackdict::IdDict) =
    for p in x; params_internal(p, ps, stackdict); end

params_internal(x::Module, ps::Vector{Param}, stackdict::IdDict) = return

params_internal(x::String, ps::Vector{Param}, stackdict::IdDict) = return

function params_internal(x::Core.SimpleVector, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for p in x; params_internal(p, ps, stackdict); end
end

function params_internal(@nospecialize(x), ps::Vector{Param}, stackdict::IdDict)
    T = typeof(x)::DataType
    nf = nfields(x)
    (isbitstype(T) || nf == 0) && return
    if haskey(stackdict, x)
        return
    end
    if ismutable(x)
        stackdict[x] = true
    end
    for i in 1:nf
        if isdefined(x,i)
            params_internal(getfield(x,i), ps, stackdict)
        end
    end
end

function params_internal(x::Array, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    _params_array_t(x, eltype(x), ps, stackdict)
end

function _params_array_t(@nospecialize(x), T, ps::Vector{Param}, stackdict::IdDict)
    stackdict[x] = true
    if isbitstype(T)
        return
    end
    for i = 1:(length(x)::Int)
        if ccall(:jl_array_isassigned, Cint, (Any, Csize_t), x, i-1) != 0
            xi = ccall(:jl_arrayref, Any, (Any, Csize_t), x, i-1)
            if !isbits(xi)
                params_internal(xi, ps, stackdict)
            end
        end
    end
end

function params_internal(x::Union{Dict,IdDict}, ps::Vector{Param}, stackdict::IdDict)
    if haskey(stackdict, x)
        return
    end
    stackdict[x] = true
    for (k, v) in x
        params_internal(k, ps, stackdict)
        params_internal(v, ps, stackdict)
    end
end

