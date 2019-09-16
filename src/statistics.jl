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

@primitive mean(x;d...),dy  (dy.*one.(x).*length(dy)./length(x))
@primitive mean(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x).*length(dy)./length(x))
@primitive mean(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x).*length(dy)./length(x))

function varm(x::Value, m; corrected::Bool=true, dims=:)
    s = sum(abs2, x .- m; dims=dims)
    r = length(x) ÷ length(s) - Int(corrected)
    s ./ r
end

function var(x::Value; corrected::Bool=true, mean=nothing, dims=:)
    varm(x, something(mean, Statistics.mean(x,dims=dims)); corrected=corrected, dims=dims)
end

std(x::Value, args...; kws...) = sqrt.(var(x, args...; kws...))
stdm(x::Value, args...; kws...) = sqrt.(varm(x, args...; kws...))
