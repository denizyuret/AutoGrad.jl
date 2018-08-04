# Print out functions defined in a .jl file
# Usage: julia functions.jl source.jl

function functions(ex::Expr,s::Set)
    if ((ex.head == :function || ex.head == :(=)) && isa(ex.args[1],Expr) && ex.args[1].head == :call)
        fn = ex.args[1].args[1]
        isa(fn, Expr) && fn.head == :curly && (fn = fn.args[1])
        if !in(fn,s)
            push!(s,fn)
            if !isa(fn,Symbol)
                println("# Not a symbol: $fn")
            elseif !isdefined(Main,fn)
                println("# Not exported: $fn")
            else
                println("# $fn")
            end
        end
    else
        for a in ex.args
            if isa(a, Expr)
                functions(a,s)
            end
        end
    end
end

ex = Meta.parse("module Foo\n"*read(ARGS[1],String)*"\nend\n")
functions(ex,Set())
