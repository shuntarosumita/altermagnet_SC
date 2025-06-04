using LinearAlgebra
using Optim
using Printf
const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

mutable struct Parameters
    t1::Float64
    t2::Float64
    α::Float64
    h_dir::String
    h::Float64
    n_fill::Float64
    T::Float64
    SC_type::String
    V::Float64
    Q::Vector{Float64}
    nk1::Int
    nk2::Int
    nk3::Int
    nwan::Int
    eps::Float64
    it_max::Int
    savepath::String
end

function Parameters(
        t1::Float64,
        t2::Float64,
        α::Float64,
        h_dir::String,
        h::Float64,
        n_fill::Float64,
        T::Float64,
        SC_type::String,
        V::Float64,
        nk1::Int,
        nk2::Int,
        nk3::Int,
    )::Parameters

    # path of save file
    savepath::String = Printf.format(
        Printf.Format("MF_t1_%.3f_t2_%.3f_a_%.3f_h%s_%.3f_n_%.3f_T_%.4f_%s_%.3f"),
        t1, t2, α, h_dir, h, n_fill, T, SC_type, V
    )

    # other parameters
    Q::Vector{Float64} = [0.0, 0.0, 0.0]
    nwan = 4
    eps = 1e-8
    it_max = 1000

    Parameters(t1, t2, α, h_dir, h, n_fill, T, SC_type, V, Q, nk1, nk2, nk3, nwan, eps, it_max, savepath)
end

function set_normal_hamiltonian(p::Parameters)::Array{ComplexF64, 5}
    hk_0 = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)

    h_mol::Matrix{ComplexF64} = zeros(ComplexF64, p.nwan, p.nwan)
    if p.h_dir == "f100"
        h_mol = p.h .* kron(σ0, σ1)
    elseif p.h_dir == "f010"
        h_mol = p.h .* kron(σ0, σ2)
    elseif p.h_dir == "f001"
        h_mol = p.h .* kron(σ0, σ3)
    elseif p.h_dir == "f110"
        h_mol = p.h/sqrt(2) .* (kron(σ0, σ1) .+ kron(σ0, σ2))
    elseif p.h_dir == "a100"
        h_mol = p.h .* kron(σ3, σ1)
    elseif p.h_dir == "a010"
        h_mol = p.h .* kron(σ3, σ2)
    elseif p.h_dir == "a001"
        h_mol = p.h .* kron(σ3, σ3)
    elseif p.h_dir == "a110"
        h_mol = p.h/sqrt(2) .* (kron(σ3, σ1) .+ kron(σ3, σ2))
    end

    # define Hamiltonian
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::Float64 = (2π*(ik1-1)) / p.nk1 + p.Q[1]
        k2::Float64 = (2π*(ik2-1)) / p.nk2 + p.Q[2]

        ### hopping terms
        f1::ComplexF64 = (1 + cis(k1)) * (1 + cis(-k2))
        h_hop::Matrix{ComplexF64} = (
            - p.t1 .* (real(f1) .* kron(σ1, σ0) .- imag(f1) .* kron(σ2, σ0))
            .- 2p.t2 * cos(k1) * cos(k2) .* kron(σ0, σ0)
            .- 2p.t2 * sin(k1) * sin(k2) .* kron(σ3, σ0)
        )

        ### SOC terms
        gx::ComplexF64 = 0.5im * (1 + cis(k1)) * (1 - cis(-k2))
        gy::ComplexF64 = 0.5im * (1 - cis(k1)) * (1 + cis(-k2))
        h_SOC::Matrix{ComplexF64} = p.α .* (
            real(gx) .* kron(σ1, σ1) .- imag(gx) .* kron(σ2, σ1)
            .+ real(gy) .* kron(σ1, σ2) .- imag(gy) .* kron(σ2, σ2)
        )

        hk_0[ik1, ik2, ik3, :, :] .= h_hop .+ h_SOC .+ h_mol
    end

    hk_0
end

function set_ϕ(p::Parameters)
    # set the form of order parameter
    ϕ = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    if p.SC_type == "s"
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            ϕ[ik1, ik2, ik3, :, :] .= kron(σ0, im .* σ2)
        end
    elseif p.SC_type == "d"
        δ::Float64 = 0.0
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            k1::Float64 = (2π*(ik1-1)) / p.nk1
            k2::Float64 = (2π*(ik2-1)) / p.nk2
            ΔfB2 = 1.0 - cis(k1) + cis(k1-k2) - cis(-k2)
            ϕ[ik1, ik2, ik3, :, :] .= (
                real(ΔfB2) .* kron(σ1, im .* σ2) .- imag(ΔfB2) .* kron(σ2, im .* σ2) # singlet: dxy τ{x,y}
                .+ δ * (sin(k1) * sin(k2)) .* kron(σ0, im .* σ2) # singlet: dxy τ0
            )
        end
    end

    ## nomalization of the order parameter
    norm2 = 0.0
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        norm2 += @views tr(
            ϕ[ik1, ik2, ik3, :, :]' * ϕ[ik1, ik2, ik3, :, :]
        ) / (2 * p.nk1 * p.nk2 * p.nk3)
    end
    ϕ ./= sqrt(norm2)

    ϕ
end

function set_μ(p::Parameters, ek::Array{Float64, 4})
    # Set electron number using Brent method
    res = optimize(
        μ -> (calc_electron_density(p, ek, μ) - p.n_fill)^2,
        3*minimum(ek), 3*maximum(ek), rel_tol=1e-4, Brent()
    )
    Optim.minimizer(res)[1]
end

function calc_electron_density(p::Parameters, ek::Array{Float64, 4}, μ::Float64)
    E = fill(one(Float64), size(ek)...)
    (2 / (p.nk1 * p.nk2 * p.nk3 * p.nwan)) * sum(E ./ (E .+ exp.((ek .- μ) ./ p.T)))
end


mutable struct Mesh
    prmt::Parameters
    η::ComplexF64
    ϕ::Array{ComplexF64, 5}
    hk_0::Array{ComplexF64, 5}
    ek_0::Array{Float64, 4}
    uk_0::Array{ComplexF64, 5}
    μ::Float64
end

function Mesh(p::Parameters)::Mesh
    η::Float64 = 0.0
    ϕ = set_ϕ(p)

    # set normal Hamiltonian
    hk_0 = set_normal_hamiltonian(p)
    ek_0 = Array{Float64, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan)
    uk_0 = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        tmpe, uk_0[ik1, ik2, ik3, :, :] = eigen(@view(hk_0[ik1, ik2, ik3, :, :]))
        maximum(abs.(imag.(tmpe))) > 1e-12 && error("non-hermitian!")
        ek_0[ik1, ik2, ik3, :] .= real.(tmpe)
    end

    # set chemical potential
    μ::Float64 = set_μ(p, ek_0)

    Mesh(p, η, ϕ, hk_0, ek_0, uk_0, μ)
end

function set_MF_hamiltonian(m::Mesh)
    p::Parameters = m.prmt
    hk_MF = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, 2p.nwan, 2p.nwan)

    # define Hamiltonian
    μmat = diagm(fill(m.μ, p.nwan))
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        ### diagonal block
        hk_MF[ik1, ik2, ik3, 1:p.nwan, 1:p.nwan] .= @view(m.hk_0[ik1, ik2, ik3, :, :]) .- μmat
        hk_MF[ik1, ik2, ik3, p.nwan+1:end, p.nwan+1:end] .= - @views(
            transpose(m.hk_0[mod(2-ik1, 1:p.nk1), mod(2-ik2, 1:p.nk2), mod(2-ik3, 1:p.nk3), :, :])
        ) .+ μmat

        ### off-diagonal block
        hk_MF[ik1, ik2, ik3, 1:p.nwan, p.nwan+1:end] .= @views(m.η .* m.ϕ[ik1, ik2, ik3, :, :])
        hk_MF[ik1, ik2, ik3, p.nwan+1:end, 1:p.nwan] .= @views(conj(m.η) .* m.ϕ[ik1, ik2, ik3, :, :]')
    end

    hk_MF
end


mutable struct MeanField
    hk_MF::Array{ComplexF64, 5}
    ek_MF::Array{Float64, 4}
    uk_MF::Array{ComplexF64, 5}
end

function MeanField(m::Mesh)::MeanField
    p::Parameters = m.prmt

    # set BdG Hamiltonian
    hk_MF = set_MF_hamiltonian(m)
    ek_MF = Array{Float64, 4}(undef, p.nk1, p.nk2, p.nk3, 2p.nwan)
    uk_MF = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, 2p.nwan, 2p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        tmpe, uk_MF[ik1, ik2, ik3, :, :] = eigen(@view(hk_MF[ik1, ik2, ik3, :, :]))
        maximum(abs.(imag.(tmpe))) > 1e-12 && error("non-hermitian!")
        ek_MF[ik1, ik2, ik3, :] .= real.(tmpe)
    end

    MeanField(hk_MF, ek_MF, uk_MF)
end

function fermi(E::Real, T::Real)
    0.5 * (1 - tanh(E / (2T)))
end

function calc_order_parameter(m::Mesh, mf::MeanField)
    p::Parameters = m.prmt

    # calculation of order parameter
    η::ComplexF64 = 0.0
    for ie in 1:2p.nwan, ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        η -= fermi(mf.ek_MF[ik1, ik2, ik3, ie], p.T) * sum(
            conj(mf.uk_MF[ik1, ik2, ik3, iζ1+p.nwan, ie])
            * conj(m.ϕ[ik1, ik2, ik3, iζ2, iζ1]) * mf.uk_MF[ik1, ik2, ik3, iζ2, ie]
            for iζ1 in 1:p.nwan, iζ2 in 1:p.nwan
        )
    end

    # order parameter per unit cell
    η * p.V / (p.nk1 * p.nk2 * p.nk3)
end

function solve_MF_equation!(m::Mesh)
    p::Parameters = m.prmt

    # initial parameters
    ## 'seed' of the mean field
    η::ComplexF64 = m.η = 1.0
    η_old1::ComplexF64 = 0.0; η_old2::ComplexF64 = 0.0
    mf::MeanField = MeanField(m)

    # iteration for self-consistent solution
    count::Int64 = 1
    rel_err::Float64 = abs(η - 2η_old1 + η_old2) / abs(η)
    while rel_err > p.eps && abs(η) > 1e-8
        ### break the loop if the count exceeds it_max
        if count >= p.it_max
            println("rel_err=$rel_err after $(p.it_max) loops -> self-consistent calculation stop!")
            println("$count: η = $η; not converge")
            break
        end

        ### determine the next value of the order parameter
        #### Steffensen method
        η_old1 = m.η = calc_order_parameter(m, mf)
        mf = MeanField(m) # recalculate the BdG Hamiltonian
        η_old2 = m.η = calc_order_parameter(m, mf)
        mf = MeanField(m)
        η -= (η_old1 - η)^2 / (η - 2η_old1 + η_old2)
        m.η = η

        rel_err = abs(η - 2η_old1 + η_old2) / abs(η)

        ### output the convergence process
        count % 101 == 0 && println("$count: η = $η")
        count += 1
    end
    # while rel_err > p.eps && abs(η) > 1e-8
    #     ### break the loop if the count exceeds it_max
    #     if count >= p.it_max
    #         println("rel_err=$rel_err after $(p.it_max) loops -> self-consistent calculation stop!")
    #         println("$count: η = $η; not converge")
    #         break
    #     end

    #     ### determine the next value of the order parameter
    #     η_old1 = η
    #     η = m.η = calc_order_parameter(m, mf)
    #     mf = MeanField(m) # recalculate the BdG Hamiltonian

    #     rel_err = abs(η - η_old1) / abs(η)

    #     ### output the convergence process
    #     count % 101 == 0 && println("$count: η = $η")
    #     count += 1
    # end

    # output the converged solution
    count < p.it_max && println("$count: η = $η; converge")

    m.η = η
end

function calc_free_energy(m::Mesh)
    p::Parameters = m.prmt
    mf::MeanField = MeanField(m)

    f::Float64 = 0.0
    for ie in p.nwan+1:2p.nwan, ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        f += - p.T * log1p(exp(- mf.ek_MF[ik1, ik2, ik3, ie] / p.T)) - 0.5 * mf.ek_MF[ik1, ik2, ik3, ie]
    end
    f /= p.nk1 * p.nk2 * p.nk3
    f += abs2(m.η) / (2 * p.V)

    f
end

function main()
    length(ARGS) < 12 && error("usage: julia main.jl t1 t2 α h_dir h n_fill T SC_type V nk1 nk2 nk3")
    t1::Float64 = parse(Float64, ARGS[1])
    t2::Float64 = parse(Float64, ARGS[2])
    α::Float64 = parse(Float64, ARGS[3])
    h_dir::String = ARGS[4]
    h::Float64 = parse(Float64, ARGS[5])
    n_fill::Float64 = parse(Float64, ARGS[6])
    T::Float64 = parse(Float64, ARGS[7])
    SC_type::String = ARGS[8]
    V::Float64 = parse(Float64, ARGS[9])
    nk1::Int = parse(Int, ARGS[10])
    nk2::Int = parse(Int, ARGS[11])
    nk3::Int = parse(Int, ARGS[12])
    p = Parameters(t1, t2, α, h_dir, h, n_fill, T, SC_type, V, nk1, nk2, nk3)
    m = Mesh(p)

    # calculate free energy for the normal state
    f_n::Float64 = calc_free_energy(m)
    println("μ = $(m.μ)")
    println("f_n = $f_n")

    ##### test begin #####
    # p.Q = [0.5, 0.0, 0.0]
    # m = Mesh(p)
    # println()
    # println("##### Q = $(p.Q) #####")
    # println("μ = $(m.μ)")

    # m.η = 0.1
    # mf = MeanField(m)
    # plt = scatter(
    #     [(2π*(ik1-1)) / p.nk1 for ik1 in 1:p.nk1], mf.ek_MF[:, 1, 1, :],
    #     legend=false, markersize=2, markercolor=:blue
    # )
    # savefig(plt, "images/test.png")
    ##### test end #####

    # solve mean-field equation for each Q
    isdir("data_original") || mkdir("data_original")
    Q1s = 0:0.01:0.1
    Q2s = 0:0.01:0.1
    η = Matrix{Float64}(undef, length(Q1s), length(Q2s))
    f_s = Matrix{Float64}(undef, length(Q1s), length(Q2s))
    for iQ1 in eachindex(Q1s)
        for iQ2 in eachindex(Q2s)
            ### set mesh for given Q
            p.Q = [Q1s[iQ1], Q2s[iQ2], 0.0]
            m = Mesh(p)
            println()
            println("##### Q = $(p.Q) #####")
            println("μ = $(m.μ)")

            ### calculate free energy for the superconducting state
            solve_MF_equation!(m)
            η[iQ1, iQ2] = real(m.η)
            f_s[iQ1, iQ2] = calc_free_energy(m)
            println("f_s = $(f_s[iQ1, iQ2])")

            ### output the solution and free energy to a txt file
            open("data_original/$(p.savepath).txt", "a") do data
                println(data, "$(p.Q[1]) $(p.Q[2]) $(p.Q[3]) $(η[iQ1, iQ2]) $(f_s[iQ1, iQ2] - f_n)")
            end
        end

        open("data_original/$(p.savepath).txt", "a") do data
            println(data, "")
        end
    end

    # output the data to minimize the free energy
    if maximum(abs, η) <= 1e-8
        iQ1min = findfirst(Q1s .== 0.0)
        iQ2min = findfirst(Q2s .== 0.0)
        iQmin = CartesianIndex(iQ1min, iQ2min)
    else
        iQmin = argmin(f_s)
        iQ1min, iQ2min = Tuple(iQmin)
    end

    savepath_fmin::String = Printf.format(
        Printf.Format("fmin_t1_%.3f_t2_%.3f_h%s_n_%.3f_T_%.4f_%s_%.3f"),
        t1, t2, h_dir, n_fill, T, SC_type, V
    )
    open("data_original/$(savepath_fmin).txt", "a") do data_fmin
        println(data_fmin, "$(h) $(α) $(Q1s[iQ1min]) $(Q2s[iQ2min]) $(η[iQmin]) $(f_s[iQmin] - f_n)")
    end

    nothing
end

main()
