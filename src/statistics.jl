# Completed: 6/13

import Statistics: mean, var, std, realXcY

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

realXcY(x::Rec, y::Rec) = x*y
@primitive mean(x;d...),dy  (dy.*one.(x).*(length(dy)/length(x)))
@primitive mean(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x).*(length(dy)/length(x)))
@primitive mean(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x).*(length(dy)/length(x)))
