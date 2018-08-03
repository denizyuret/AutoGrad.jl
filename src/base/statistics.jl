import Base: var, std

function var(x::Rec, dims; mean=Base.mean(x, dims), corrected=true)
    s = sum(abs2, x .- mean, dims)
    a = length(x) ÷ length(s) 
    corrected ? s ./ (a-1) : s ./ a  
end

function var(x::Rec; mean=Base.mean(x), corrected=true)
    s = sum(abs2, x .- mean)
    a = length(x) ÷ length(s) 
    corrected ? s / (a-1) : s / a  
end

addtest(:var, rand(2,3))
addtest(:var, rand(2,3), 1)
addtest(:var, rand(2,3), (1,2))

std(x::Rec, args...; kws...) = sqrt.(var(x, args...; kws...))
addtest(:std, rand(2,3))
addtest(:std, rand(2,3), 1)
addtest(:std, rand(2,3), (1,2))
