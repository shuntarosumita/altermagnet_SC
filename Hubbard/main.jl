using Printf
using HDF5
using LinearAlgebra
using Optim
using FFTW
using FLoops
using TensorCore
using SparseIR
import SparseIR: Statistics
import SparseIR: valueim
using Dates
const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

include("parameters.jl")
include("mesh.jl")
include("gfunction.jl")
include("eliashberg.jl")
include("kpath_extract.jl")

function main()
    length(ARGS) < 17 && error("usage: julia main.jl system SC_type h h_load h_dir α α_load n U U_load T round_it mode nk1 nk2 nk3 ωmax (data_dir)")
    system::String = ARGS[1]
    SC_type::String = ARGS[2]
    h::Float64 = parse(Float64, ARGS[3])
    h_load::Float64 = parse(Float64, ARGS[4])
    h_dir::String = ARGS[5]
    α::Float64 = parse(Float64, ARGS[6])
    α_load::Float64 = parse(Float64, ARGS[7])
    n_fill::Float64 = parse(Float64, ARGS[8])
    U::Float64 = parse(Float64, ARGS[9])
    U_load::Float64 = parse(Float64, ARGS[10])
    T::Float64 = parse(Float64, ARGS[11])
    round_it::Int64 = parse(Int64, ARGS[12])
    mode::String = ARGS[13]
    nk1::Int64 = parse(Int64, ARGS[14])
    nk2::Int64 = parse(Int64, ARGS[15])
    nk3::Int64 = parse(Int64, ARGS[16])
    ωmax::Float64 = parse(Float64, ARGS[17])
    data_dir::String = (length(ARGS) < 18 ? "./" : ARGS[18])
    println("nthreads = ", Threads.nthreads())

    ### tetragonal condition
    nk1 != nk2 && error("nk1 = nk2 should be satisfied due to the tetragonal symmetry")

    ### RPA
    if mode == "RPA"
        println("automatically set U = 1 for RPA mode")
        U = U_load = 1.0
    end

    ### Initiate parameters -------------------------------------------------------
    start = now()
    p = Parameters(
        system, SC_type, h, h_dir, α, n_fill, U, T, round_it, mode;
        nk1, nk2, nk3, ωmax,
        h_load=h_load, α_load=α_load, U_load=U_load,
        data_dir=data_dir
    )
    save_Parameters(p)
    open(p.Logstr, "a") do log
        println(log, "##################################################")
        println(
            log,
            @sprintf(
                "Parameter set: h%s = %.3f, α = %.3f, n = %.3f, U = %.3f, T = %.4f",
                p.h_dir, p.h, p.α, p.n_fill, p.U, p.T
            )
        )
        println(
            log,
            "Elapsed time - parameter init: ",
            (now() - start).value / 1000, "s"
        )
    end

    ### Load mesh -----------------------------------------------------------------
    t_mset = now()
    m = Mesh(p)
    open(p.Logstr, "a") do log
        println(log, "fnω, fnτ, iω0_f: $(m.fnω), $(m.fnτ), $(m.iω0_f)")
        println(log, "bnω, bnτ, iω0_b: $(m.bnω), $(m.bnτ), $(m.iω0_b)")
        println(log, "emin = $(m.emin), emax = $(m.emax); μ = $(m.μ)")
        println(
            log,
            "Elapsed time - Mesh set (tot | module): ",
            (now() - start).value / 1000, "s | ", (now() - t_mset).value / 1000, "s"
        )
    end

    ### The case of skipping Green function calculation
    if minimum(abs.(m.ek)) > 1e-2
        open(p.Logstr, "a") do log
            println(log, "Insulating state => Not calculating Green function.")
            println(log, "##################################################")
        end
        return nothing
    end

    ### Calculate full Green function ---------------------------------------------
    t_gset = now()
    g = Gfunction(m) # initialize Gfunction
    if p.mode == "FLEX"
        FLEXcheck = solve_FLEX!(m, g)
    end
    save_Gfunctions(m, g)
    open(p.Logstr, "a") do log
        println(
            log,
            "Elapsed time - Gfunction calc (tot | module): ",
            (now() - start).value / 1000, "s | ", (now() - t_gset).value / 1000, "s"
        )
    end

    ### Extract quantities along HS path in BZ ------------------------------------
    kpath_extract(m, g)

    ### The case of skipping Eliashberg calculation
    if p.SC_type == "skip"
        open(p.Logstr, "a") do log
            println(log, "SC_type = skip => Not calculating eliashberg equation.")
            println(log, "##################################################")
        end
        return nothing
    elseif p.mode == "FLEX"
        # U, Σ convergence
        if FLEXcheck === missing
            open(p.Logstr, "a") do log
                println(log, "Not calculating eliashberg equation.")
                println(log, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                println(log, "##################################################")
            end
            open(p.Logerrstr, "a") do logerr
                println(logerr, "$(p.err_str_begin) => eliashberg skipped.")
            end
            return missing
        end
    elseif p.mode == "RPA"
        p.U *= 0.98 / max_eigval_χU(m, g)
        m.U_mat = set_interaction(p)
        set_χiωk!(m, g)

        open(p.Logstr, "a") do log
            println(log, "### Set new U = $(p.U) to satisfy max(χU) = 0.98")
        end
    end

    ### Calculate SC parameter --------------------------------------------------
    open(p.Logstr, "a") do log
        println(log, "Move to SC calculation.")
    end
    Qdiv = 32
    # Qrange = [0 0;]
    Qrange = hcat(
        [ik ÷ (nk2÷Qdiv+1) for ik in 0:((nk1÷Qdiv+1)*(nk2÷Qdiv+1)-1)],
        [ik % (nk2÷Qdiv+1) for ik in 0:((nk1÷Qdiv+1)*(nk2÷Qdiv+1)-1)]
    )

    e = Eliashberg(m, g, [0, 0, 0])
    for (iQ1, iQ2) in eachrow(Qrange)
        iQ::Vector{Int64} = [iQ1, iQ2, 0]
        open(p.Logstr, "a") do log
            println(log, "Eliashberg calculation for irrep = $SC_type and iQ = $iQ")
        end

        t_eset = now()
        e.iQ .= iQ
        solve_Eliashberg!(m, g, e)
        save_Eliashberg(m, g, e)
        open(p.Logstr, "a") do log
            println(
                log,
                @sprintf(
                    "Done: h%s = %.3f | α = %.3f | n = %.3f | U = %.3f | T = %.4f (%.3f K) | λ = %.6f",
                    p.h_dir, p.h, p.α, p.n_fill, p.U, p.T, p.T*1.16045*10^4, e.λ
                )
            )
            println(
                log,
                "Elapsed time - Eliashberg calc (tot | module): ",
                (now() - start).value / 1000, "s | ", (now() - t_eset).value / 1000, "s"
            )
        end
        if iQ2 == nk2÷Qdiv
            open(p.SC_EV_path, "a") do file
                println(file, "")
            end
        end

        ### Extract quantities along HS path in BZ ------------------------------------
        if e.iQ == [0, 0, 0]
            kpath_extract(m, e)
        end
    end

    open(p.Logstr, "a") do log
        println(log, "##################################################")
    end

    return nothing
end

main()
