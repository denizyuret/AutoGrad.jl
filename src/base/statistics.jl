if VERSION < v"0.7.0-DEV.4064"
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
else
    function var(x::Rec; dims = :, mean=Base.mean(x, dims = dims), corrected=true)
        s = sum(abs2, x .- mean, dims = dims)
        a = length(x) ÷ length(s) 
        corrected ? s ./ (a-1) : s ./ a  
    end
end

addtest(:var, rand(2,3))
addtest(:std, rand(2,3))

if VERSION < v"0.7.0-DEV.4064"
    addtest(:var, rand(2,3), 1)
    addtest(:std, rand(2,3), 1)
    addtest(:std, rand(2,3), (1,2))
    addtest(:var, rand(2,3), (1,2))
    std(x::Rec, args...; kws...) = sqrt.(var(x, args...; kws...))
else
    addtest(:var, rand(2,3), dims = 1)
    addtest(:std, rand(2,3), dims = 1)
    addtest(:std, rand(2,3), dims = (1,2))
    addtest(:var, rand(2,3), dims = (1,2))
    std(x::Rec; kws...) = sqrt.(var(x; kws...))
end

