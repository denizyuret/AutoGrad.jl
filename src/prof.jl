# Inspired by TimerOutputs.jl, which breaks with stack overflow in AutoGrad.

using Base: gc_bytes, time_ns
export profreset!, proftable

type TimeData
    ncalls::Int64
    time::Int64
    allocs::Int64
end

TimeData() = TimeData(0, 0, 0)

PROFHASH=Dict{String,TimeData}()

function profreset!()
    global PROFHASH
    empty!(PROFHASH)
end

function proftable()
    global PROFHASH
    time = allocs = 0
    for (k,v) in PROFHASH
        time += v.time
        allocs += v.allocs
    end
    for (k,v) in sort(collect(PROFHASH), rev=true, by=(p->p[2].time))
        @printf "%d\t%.2g (%.1f%%)\t%.2g (%.1f%%)\t%s\n" v.ncalls v.time 100*v.time/time v.allocs 100*v.allocs/allocs k
    end
end

macro prof(label,ex)
    quote
        local td = get!(TimeData,PROFHASH,$(esc(label)))
        local b₀ = gc_bytes()
        local t₀ = time_ns()
        local val = $(esc(ex))
        td.time += time_ns() - t₀
        td.allocs += gc_bytes() - b₀
        td.ncalls += 1
        val
    end
end
