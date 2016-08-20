function functions(ex::Expr,s::Set)
    if ((ex.head == :function || ex.head == :(=)) && isa(ex.args[1],Expr) && ex.args[1].head == :call)
        fn = ex.args[1].args[1]
        isa(fn, Expr) && fn.head == :curly && (fn = fn.args[1])
        if !in(fn,s)
            push!(s,fn)
            if !isa(fn,Symbol)
                println("# $fn: Not a symbol")
            elseif !isdefined(fn)
                println("# $fn: Not exported")
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

ex = parse("module Foo\n"*readall(ARGS[1])*"\nend\n")
functions(ex,Set())
