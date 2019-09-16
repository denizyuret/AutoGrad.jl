# Completed: 8/13

import Statistics: mean, var, std, varm, stdm

# TODO:
# cor
# cov
# mean done
# mean! overwrites.
# median
# median! overwrites.
# middle
# quantile
# quantile! overwrites.
# std done.
# stdm done.
# var done.
# varm done.

@primitive mean(x;d...),dy  (dy.*one.(x)./(length(x)÷length(dy)))
@primitive mean(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x)./(length(x)÷length(dy)))
@primitive mean(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x)./(length(x)÷length(dy)))

std(x::Value, args...; kws...) = sqrt.(var(x, args...; kws...))
stdm(x::Value, args...; kws...) = sqrt.(varm(x, args...; kws...))

var(x::Value; corrected::Bool=true, mean=nothing, dims=:)=_varm(x, something(mean, Statistics.mean(x,dims=dims)); corrected=corrected, dims=dims)
varm(x::Value, m; corrected::Bool=true, dims=:)=_varm(x, m; corrected=corrected, dims=dims)

function _varm(x, m; corrected::Bool=true, dims=:)
    s = sum(abs2, x .- m; dims=dims)
    r = length(x) ÷ length(s) - Int(corrected)
    s ./ r
end

# I don't think making _varm a primitive improves performance, replaces a single sum op with a single _varm op.
# @primitive _varm(x, m; corrected::Bool=true, dims=:),dy  (dy.*(x.-m).*(2/(length(x)÷length(dy)-Int(corrected))))  nothing
# @primitive sum(x;d...),dy  (dy.*one.(x))
# @primitive sum(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x))
# @primitive sum(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x))
