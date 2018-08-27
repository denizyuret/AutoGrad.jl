# Completed: 6/13

import Statistics: mean, var, std

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
# stdm
# var done.
# varm

@primitive mean(x;d...),dy  (dy.*one.(x).*length(dy)./length(x))
@primitive mean(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x).*length(dy)./length(x))
@primitive mean(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x).*length(dy)./length(x))

function var(x::Value; dims=:, mean=mean(x, dims=dims), corrected=true)
    s = sum(abs2, x .- mean, dims=dims)
    a = length(x) ÷ length(s) 
    corrected ? s ./ (a-1) : s ./ a  
end

std(x::Value, args...; kws...) = sqrt.(var(x, args...; kws...))
