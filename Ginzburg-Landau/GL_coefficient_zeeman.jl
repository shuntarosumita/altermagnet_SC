using LinearAlgebra
using GenericLinearAlgebra
using Optim
using Printf
using StatsBase
setprecision(113)
const BigComp = Complex{BigFloat}
const σ0 = BigComp[1 0; 0 1]
const σ1 = BigComp[0 1; 1 0]
const σ2 = BigComp[0 -1im; 1im 0]
const σ3 = BigComp[1 0; 0 -1]

# vectors for numerical derivative
const γ_diff::Vector{Vector{BigFloat}} = [
    [0, 0, 1, 0, 0],
    [1, -8, 0, 8, -1] ./ 12,
    [-1, 16, -30, 16, -1] ./ 12,
    [-1, 2, 0, -2, 1] ./ 2,
    [1, -4, 6, -4, 1]
]

mutable struct Parameters
    t1::BigFloat
    t2::BigFloat
    α::BigFloat
    h::BigFloat
    h_dir::String
    n_fill::BigFloat
    μ::BigFloat
    T::BigFloat
    SC_type::String
    nk1::Int
    nk2::Int
    nk3::Int
    nk::Int
    nwan::Int
    savepath::String
end

function Parameters(
        t1::BigFloat,
        t2::BigFloat,
        α::BigFloat,
        h::BigFloat,
        h_dir::String,
        n_fill::BigFloat,
        T::BigFloat,
        SC_type::String,
        nk1::Int,
        nk2::Int,
        nk3::Int
    )::Parameters

    # path of save file
    savepath::String = Printf.format(
        Printf.Format("GL_zeeman_t1_%.3f_t2_%.3f_h%s_a_%.3f_n_%.3f_T_%.4f_%s"),
        t1, t2, h_dir, α, n_fill, T, SC_type
    )
    nwan = 2
    nk = nk1 * nk2 * nk3

    Parameters(t1, t2, α, h, h_dir, n_fill, 0.0, T, SC_type, nk1, nk2, nk3, nk, nwan, savepath)
end

function calc_hamiltonian(p::Parameters)::Array{BigComp, 5}
    hk = Array{BigComp, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    h_zeeman::Matrix{BigComp} = zeros(BigComp, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::BigFloat = (2π*(ik1-1)) / p.nk1
        k2::BigFloat = (2π*(ik2-1)) / p.nk2

        ### hopping terms
        h_hop::Matrix{BigComp} = (
            - 2p.t1 * (cos(k1) + cos(k2)) - 2p.t2 * (cos(2k1) + cos(2k2))
            #- 4p.t2 * cos(k1) * cos(k2)
        ) .* σ0

        ### SOC terms (Rashba)
        h_SOC::Matrix{BigComp} = p.α .* (sin(k2) .* σ1 .- sin(k1) .* σ2)

        ### Zeeman term
        if p.h_dir == "f100"
            h_zeeman = - p.h .* σ1
        elseif p.h_dir == "f010"
            h_zeeman = - p.h .* σ2
        elseif p.h_dir == "f001"
            h_zeeman = - p.h .* σ3
        elseif p.h_dir == "f110"
            h_zeeman = - p.h/sqrt(2) .* (σ1 .+ σ2)
        elseif p.h_dir == "a100"
            h_zeeman = - p.h * (cos(k1) - cos(k2)) .* σ1
        elseif p.h_dir == "a010"
            h_zeeman = - p.h * (cos(k1) - cos(k2)) .* σ2
        elseif p.h_dir == "a001"
            h_zeeman = - p.h * (cos(k1) - cos(k2)) .* σ3
        elseif p.h_dir == "a110"
            h_zeeman = - p.h/sqrt(2) * (cos(k1) - cos(k2)) .* (σ1 .+ σ2)
        elseif p.h_dir == "ad100"
            h_zeeman = - p.h * (cos(2k1) - cos(2k2)) .* σ1
        elseif p.h_dir == "ad001"
            h_zeeman = - p.h * (cos(2k1) - cos(2k2)) .* σ3
        elseif p.h_dir == "dxy001"
            h_zeeman = - 2p.h * sin(k1) * sin(k2) .* σ3
        end

        hk[ik1, ik2, ik3, :, :] .= h_hop .+ h_SOC .+ h_zeeman
    end

    hk
end

function calc_hamiltonian_spin_resolved(p::Parameters)
    ek_u = Array{BigFloat, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan÷2)
    ek_d = Array{BigFloat, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan÷2)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::BigFloat = (2π*(ik1-1)) / p.nk1
        k2::BigFloat = (2π*(ik2-1)) / p.nk2

        ### hopping terms
        h_hop = - 2p.t1 * (cos(k1) + cos(k2)) - 4p.t2 * cos(k1) * cos(k2)

        ### Zeeman term
        if p.h_dir == "f100" || p.h_dir == "f010" || p.h_dir == "f001" || p.h_dir == "f110"
            h_zeeman = - p.h
        elseif p.h_dir == "a100" || p.h_dir == "a010" || p.h_dir == "a001" || p.h_dir == "a110"
            h_zeeman = - p.h * (cos(k1) - cos(k2))
        elseif p.h_dir == "ad100" || p.h_dir == "ad010" || p.h_dir == "ad001" || p.h_dir == "ad110"
            h_zeeman = - p.h * (cos(2k1) - cos(2k2))
        elseif p.h_dir == "dxy001"
            h_zeeman = - 2p.h * sin(k1) * sin(k2)
        end

        ek_u[ik1, ik2, ik3, 1] = h_hop + h_zeeman
        ek_d[ik1, ik2, ik3, 1] = h_hop - h_zeeman
    end

    ek_u, ek_d
end

function set_μ(p::Parameters, ek::Array{BigFloat, 4})
    # Set electron number using Brent method
    ## n_0 is per orbital
    n_0::BigFloat = p.n_fill

    res = optimize(
        μ -> (calc_electron_density(p, ek, μ) - n_0)^2,
        3*minimum(ek), 3*maximum(ek), rel_tol=parse(BigFloat, "1e-4"), Brent()
    )
    Optim.minimizer(res)[1]
end

function calc_electron_density(p::Parameters, ek::Array{BigFloat, 4}, μ::BigFloat)
    E = fill(one(BigFloat), size(ek)...)
    (2 / (p.nk * p.nwan)) * sum(E ./ (E .+ exp.((ek .- μ) ./ p.T)))
end

function fermi(E::Real, T::Real, diff::Int=0)
    f = (1 - tanh(E / (2T))) / 2
    if diff == 0
        return f
    elseif diff == 1
        return f * (f - 1) / T
    elseif diff == 2
        return f * (f - 1) * (2f - 1) / T^2
    elseif diff == 3
        return f * (f - 1) * (6f^2 - 6f + 1) / T^3
    end
end

function Matsubara_sum(T::Real, z1::Real, z2::Real)
    if z1 != z2
        return (fermi(z1, T) - fermi(z2, T)) / (z1 - z2)
    else
        return fermi(z1, T, 1)
    end
end

function Matsubara_sum(T::Real, z1::Real, z2::Real, z3::Real)
    z_c = countmap([z1, z2, z3])
    z_c_1 = [k for (k, v) in z_c if v == 1]

    if length(z_c_1) == 3
        # z1, z2, and z3 are all different
        return (
            + fermi(z1, T) / ((z1 - z2) * (z1 - z3))
            + fermi(z2, T) / ((z2 - z3) * (z2 - z1))
            + fermi(z3, T) / ((z3 - z1) * (z3 - z2))
        )
    elseif length(z_c_1) == 1
        # two of the three elements (z1, z2, z3) are the same
        ## e.g. z1 == z2 != z3
        z_c_2 = [k for (k, v) in z_c if v == 2]
        return (
            - fermi(z_c_2[1], T, 1) / (z_c_1[1] - z_c_2[1])
            + (fermi(z_c_1[1], T) - fermi(z_c_2[1], T)) / (z_c_1[1] - z_c_2[1])^2
        )
    else
        # z1 == z2 == z3
        return fermi(z1, T, 2) / 2
    end
end

function Matsubara_sum(T::Real, z1::Real, z2::Real, z3::Real, z4::Real)
    z_c = countmap([z1, z2, z3, z4])
    z_c_1 = [k for (k, v) in z_c if v == 1]

    if length(z_c_1) == 4
        # z1, z2, z3, and z4 are all different
        return (
            + fermi(z1, T) / ((z1 - z2) * (z1 - z3) * (z1 - z4))
            + fermi(z2, T) / ((z2 - z3) * (z2 - z4) * (z2 - z1))
            + fermi(z3, T) / ((z3 - z4) * (z3 - z1) * (z3 - z2))
            + fermi(z4, T) / ((z4 - z1) * (z4 - z2) * (z4 - z3))
        )
    elseif length(z_c_1) == 2
        # two of the four elements (z1, z2, z3, z4) are the same, and the others are different
        ## e.g. z1 == z2, z1 != z3, z1 != z4, z3 != z4
        z_c_2 = [k for (k, v) in z_c if v == 2]
        return (
            fermi(z_c_2[1], T, 1) / ((z_c_1[1] - z_c_2[1]) * (z_c_1[2] - z_c_2[1]))
            + (fermi(z_c_1[1], T) - fermi(z_c_2[1], T)) / ((z_c_1[1] - z_c_2[1])^2 * (z_c_1[1] - z_c_1[2]))
            + (fermi(z_c_1[2], T) - fermi(z_c_2[1], T)) / ((z_c_1[2] - z_c_2[1])^2 * (z_c_1[2] - z_c_1[1]))
        )
    elseif length(z_c_1) == 1
        # three of the four elements (z1, z2, z3, z4) are the same
        ## e.g. z1 == z2 == z3 != z4
        z_c_3 = [k for (k, v) in z_c if v == 3]
        return (
            - fermi(z_c_3[1], T, 2) / (2 * (z_c_1[1] - z_c_3[1]))
            - fermi(z_c_3[1], T, 1) / (z_c_1[1] - z_c_3[1])^2
            + (fermi(z_c_1[1], T) - fermi(z_c_3[1], T)) / (z_c_1[1] - z_c_3[1])^3
        )
    else
        z_c_2 = [k for (k, v) in z_c if v == 2]
        if length(z_c_2) == 2
            # two pairs in the four elements (z1, z2, z3, z4) that have the same value exist
            ## e.g. z1 == z2 != z3 == z4
            return (
                (fermi(z_c_2[1], T, 1) + fermi(z_c_2[2], T, 1)) / (z_c_2[1] - z_c_2[2])^2
                - 2 * (fermi(z_c_2[1], T) - fermi(z_c_2[2], T)) / (z_c_2[1] - z_c_2[2])^3
            )
        else
            # z1 == z2 == z3 == z4
            return fermi(z1, T, 3) / 6
        end
    end
end

function calc_velocity_spin_resolved(p::Parameters)
    vk_u = Array{BigFloat, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan÷2, 3)
    vk_d = Array{BigFloat, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan÷2, 3)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::BigFloat = (2π*(ik1-1)) / p.nk1
        k2::BigFloat = (2π*(ik2-1)) / p.nk2

        ### derivative from hopping terms
        v_hop = [
            2p.t1 * sin(k1) + 4p.t2 * sin(k1) * cos(k2),
            2p.t1 * sin(k2) + 4p.t2 * cos(k1) * sin(k2),
            0.0
        ]

        ### derivative from Zeeman term
        v_zeeman = zeros(BigFloat, 3)
        if p.h_dir == "a100" || p.h_dir == "a010" || p.h_dir == "a001" || p.h_dir == "a110"
            v_zeeman[1] = p.h * sin(k1)
            v_zeeman[2] = - p.h * sin(k2)
        elseif p.h_dir == "ad100" || p.h_dir == "ad010" || p.h_dir == "ad001" || p.h_dir == "ad110"
            v_zeeman[1] = 2 * p.h * sin(2k1)
            v_zeeman[2] = - 2 * p.h * sin(2k2)
        elseif p.h_dir == "dxy001"
            v_zeeman[1] = - 2p.h * cos(k1) * sin(k2)
            v_zeeman[2] = - 2p.h * sin(k1) * cos(k2)
        end

        for μ1 in 1:3
            vk_u[ik1, ik2, ik3, 1, μ1] = v_hop[μ1] + v_zeeman[μ1]
            vk_d[ik1, ik2, ik3, 1, μ1] = v_hop[μ1] - v_zeeman[μ1]
        end
    end

    vk_u, vk_d
end

function calc_GL_coefficient(p::Parameters, ϕ::AbstractArray; verbose::Bool=true)
    # calculate Hamiltonian and its eigenvalues/vectors
    hk = calc_hamiltonian(p)
    ek = Array{BigFloat, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan)
    uk = Array{BigComp, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        ek[ik1, ik2, ik3, :], uk[ik1, ik2, ik3, :, :] = eigen(@view(hk[ik1, ik2, ik3, :, :]))
    end

    # chemical potential
    p.μ = set_μ(p, ek)
    println("h = $(p.h): emin = $(minimum(ek)), emax = $(maximum(ek)); μ = $(p.μ)")
    ek .-= p.μ

    ## calculate reversed eigenvalues/vectors
    ek_rev::Array{BigFloat, 4} = reverse(
        circshift(ek, (-1, -1, -1, 0)),
        dims=(1, 2, 3)
    )
    uk_rev::Array{BigComp, 5} = conj.(
        reverse(
            circshift(uk, (-1, -1, -1, 0, 0)),
            dims=(1, 2, 3)
        )
    )

    ## band-based matrix elements
    ϕ_b = similar(ϕ)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        ϕ_b[ik1, ik2, ik3, :, :] .= @views(
            uk[ik1, ik2, ik3, :, :]' * ϕ[ik1, ik2, ik3, :, :] * uk_rev[ik1, ik2, ik3, :, :]
        )
    end

    # calculate trace for c2_n
    c2_tr::Matrix{BigComp} = zeros(BigComp, 5, 5)
    for iq2 in -2:2, iq1 in -2:2
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            ikpq1 = mod(ik1 + iq1, 1:p.nk1); ikmq1 = mod(ik1 - iq1, 1:p.nk1)
            ikpq2 = mod(ik2 + iq2, 1:p.nk2); ikmq2 = mod(ik2 - iq2, 1:p.nk2)

            ϕ_b_q = @views(
                uk[ikpq1, ikpq2, ik3, :, :]' * ϕ[ik1, ik2, ik3, :, :] * uk_rev[ikmq1, ikmq2, ik3, :, :]
            )
            for l1 in 1:p.nwan, l2 in 1:p.nwan
                c2_tr[iq1+3, iq2+3] += abs2(ϕ_b_q[l1, l2]) * Matsubara_sum(
                    p.T, ek[ikpq1, ikpq2, ik3, l1], -ek_rev[ikmq1, ikmq2, ik3, l2]
                )
            end
        end
    end
    c2_tr ./= p.nk

    # c2_0: second-order term with zeroth-order derivative
    c2_0::BigComp = c2_tr[3, 3]

    # c2_n: second-order term with nth-order derivative
    Δq1 = 2π / p.nk1
    Δq2 = 2π / p.nk2
    c2_n::Vector{Vector{BigComp}} = [
        [dot(γ_diff[n-i+2], c2_tr, γ_diff[i]) / (factorial(n) * (2im)^n * Δq1^(n-i+1) * Δq2^(i-1)) for i in 1:(n+1)]
        for n in 1:4
    ]

    ## print coefficients
    if verbose
        println("c2_0: $(real(c2_0))")
        for n in [2, 4] #1:4
            println("c2_$n:" * prod(" $(real(c2_n[n][i]))" for i in 1:(n+1)))
        end
    end

    # c4_0: fourth-order term with zeroth-order derivative
    c4_0::BigComp = parse(BigComp, "0.0")
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        for l1 in 1:p.nwan, l2 in 1:p.nwan, l3 in 1:p.nwan, l4 in 1:p.nwan
            c4_0 += (
                ϕ_b[ik1, ik2, ik3, l1, l3] * ϕ_b[ik1, ik2, ik3, l2, l4]
                * conj(ϕ_b[ik1, ik2, ik3, l1, l4] * ϕ_b[ik1, ik2, ik3, l2, l3])
                * Matsubara_sum(
                    p.T, ek[ik1, ik2, ik3, l1], ek[ik1, ik2, ik3, l2],
                    -ek_rev[ik1, ik2, ik3, l3], -ek_rev[ik1, ik2, ik3, l4]
                )
            )
        end
    end
    c4_0 /= p.nk

    ## print coefficients
    if verbose
        println("c4_0 = $(real(c4_0))")
    end

    c2_0, c2_n, c4_0
end

function calc_GL_coefficient_intraband(p::Parameters, ϕ::AbstractArray; verbose::Bool=true)
    # calculate spin-resolved Hamiltonian and its eigenvalues/vectors
    ek_u, ek_d = calc_hamiltonian_spin_resolved(p)
    ek_u .-= p.μ; ek_d .-= p.μ
    vk_u, vk_d = calc_velocity_spin_resolved(p)

    ## calculate reversed spin-down eigenvalues
    ek_d = reverse(
        circshift(ek_d, (-1, -1, -1, 0)),
        dims=(1, 2, 3)
    )
    vk_d = reverse(
        circshift(vk_d, (-1, -1, -1, 0, 0)),
        dims=(1, 2, 3)
    )

    # intraband order parameter
    ϕ_b_intra = Array{BigComp, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan÷2)
    for a1 in 1:p.nwan÷2, ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        ϕ_b_intra[ik1, ik2, ik3, a1] = sum(
            ϕ[ik1, ik2, ik3, 2ζ1-1, 2ζ2]
            for ζ2 in 1:p.nwan÷2, ζ1 in 1:p.nwan÷2
        )
    end

    # calculate trace for c2_n
    c2_tr::Array{BigComp, 4} = [
        Matsubara_sum(p.T, ek_u[ik1, ik2, ik3, a1], -ek_d[ik1, ik2, ik3, a1])
        for ik1 in 1:p.nk1, ik2 in 1:p.nk2, ik3 in 1:p.nk3, a1 in 1:p.nwan÷2
    ]
    c2_tr ./= p.nk

    # c2_n: second-order term with nth-order derivative
    Δq1 = 2π / p.nk1
    Δq2 = 2π / p.nk2
    c2_2_intra::Vector{BigComp} = zeros(BigComp, 3)
    for i in 1:(2+1)
        c2_2_intra_1::BigComp = parse(BigComp, "0.0")
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            for iq2 in -2:2, iq1 in -2:2
                ikpq1 = mod(ik1 + iq1, 1:p.nk1)
                ikpq2 = mod(ik2 + iq2, 1:p.nk2)
                for a1 in 1:p.nwan÷2
                    c2_2_intra_1 += (
                        abs2(ϕ_b_intra[ik1, ik2, ik3, a1])
                        * γ_diff[2-i+2][iq1+3] * c2_tr[ikpq1, ikpq2, ik3, a1] * γ_diff[i][iq2+3]
                    )
                end
            end
        end
        c2_2_intra_1 /= Δq1^(2-i+1) * Δq2^(i-1)

        c2_2_intra_2::BigComp = parse(BigComp, "0.0")
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            for a1 in 1:p.nwan÷2
                c2_2_intra_2 -= (
                    4 * abs2(ϕ_b_intra[ik1, ik2, ik3, a1])
                    * vk_u[ik1, ik2, ik3, a1, 1+i÷3] * vk_d[ik1, ik2, ik3, a1, 1+i÷2]
                    * Matsubara_sum(
                        p.T, ek_u[ik1, ik2, ik3, a1], ek_u[ik1, ik2, ik3, a1],
                        -ek_d[ik1, ik2, ik3, a1], -ek_d[ik1, ik2, ik3, a1]
                    )
                )
            end
        end
        c2_2_intra_2 /= p.nk

        c2_2_intra[i] = (c2_2_intra_1 + c2_2_intra_2) / (factorial(2) * (2im)^2)
    end

    ## print coefficients
    if verbose
        println("c2_2_intra:" * prod(" $(real(c2_2_intra[i]))" for i in 1:(2+1)))
    end

    # c4_0: fourth-order term with zeroth-order derivative
    c4_0_intra::BigComp = parse(BigComp, "0.0")
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        for a1 in 1:p.nwan÷2
            c4_0_intra += abs(ϕ_b_intra[ik1, ik2, ik3, a1])^4 * Matsubara_sum(
                p.T, ek_u[ik1, ik2, ik3, a1], ek_u[ik1, ik2, ik3, a1],
                -ek_d[ik1, ik2, ik3, a1], -ek_d[ik1, ik2, ik3, a1]
            )
        end
    end
    c4_0_intra /= p.nk

    ## print coefficients
    if verbose
        println("c4_0_intra:= $(real(c4_0_intra))")
    end

    c2_2_intra, c4_0_intra
end

function main()
    length(ARGS) < 13 && error("usage: julia GL_coefficient_zeeman.jl t1 t2 α h_dir n_fill T SC_type nk1 nk2 nk3 h0 dh h1")
    t1::BigFloat = parse(BigFloat, ARGS[1])
    t2::BigFloat = parse(BigFloat, ARGS[2])
    α::BigFloat = parse(BigFloat, ARGS[3])
    h::BigFloat = parse(BigComp, "0.0")
    h_dir::String = ARGS[4]
    n_fill::BigFloat = parse(BigFloat, ARGS[5])
    T::BigFloat = parse(BigFloat, ARGS[6])
    SC_type::String = ARGS[7]
    nk1::Int = parse(Int, ARGS[8])
    nk2::Int = parse(Int, ARGS[9])
    nk3::Int = parse(Int, ARGS[10])
    h0::Float64 = parse(Float64, ARGS[11])
    dh::Float64 = parse(Float64, ARGS[12])
    h1::Float64 = parse(Float64, ARGS[13])
    p = Parameters(t1, t2, α, h, h_dir, n_fill, T, SC_type, nk1, nk2, nk3)

    # set the form of order parameter
    ϕ = Array{BigComp, 5}(undef, nk1, nk2, nk3, p.nwan, p.nwan)
    if SC_type == "s"
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            ϕ[ik1, ik2, ik3, :, :] .= im .* σ2
        end
    elseif SC_type == "d"
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            k1::BigFloat = (2π*(ik1-1)) / p.nk1
            k2::BigFloat = (2π*(ik2-1)) / p.nk2
            ϕ[ik1, ik2, ik3, :, :] .= (cos(k1) - cos(k2)) * im .* σ2
        end
    elseif SC_type == "dd"
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
            k1::BigFloat = (2π*(ik1-1)) / p.nk1
            k2::BigFloat = (2π*(ik2-1)) / p.nk2
            ϕ[ik1, ik2, ik3, :, :] .= (cos(2k1) - cos(2k2)) * im .* σ2
        end
    end

    ## nomalization of the order parameter
    norm2 = parse(BigComp, "0.0")
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        norm2 += @views tr(
            ϕ[ik1, ik2, ik3, :, :]' * ϕ[ik1, ik2, ik3, :, :]
        ) / (2 * p.nk)
    end
    ϕ ./= sqrt(norm2)

    # calculate GL coefficients for each altermagnetic mean field
    hs = h0:dh:h1
    isdir("data_zeeman") || mkdir("data_zeeman")
    for ih in eachindex(hs)
        p.h = parse(BigFloat, string(hs[ih]))

        println("########## h$(p.h_dir) = $(p.h) ##########")
        c2_0, c2_n, c4_0 = calc_GL_coefficient(p, ϕ)
        # c2_2_intra, c4_0_intra = calc_GL_coefficient_intraband(p, ϕ)
        println("##############################")
        println()

        ### output the GL coefficients to a txt file
        open("data_zeeman/$(p.savepath)_c2_0.txt", "a") do data
            println(data,
                "$(p.h) $(real(c2_0))"
            )
        end
        open("data_zeeman/$(p.savepath)_c4_0.txt", "a") do data
            println(data,
                "$(p.h) $(real(c4_0))" # $(real(c4_0_intra))
            )
        end
        # open("data_zeeman/$(p.savepath)_c2_1.txt", "a") do data
        #     println(data,
        #         "$(p.h)"
        #         * prod(" $(imag(c2_n[1][i]))" for i in 1:2)
        #     )
        # end
        open("data_zeeman/$(p.savepath)_c2_2.txt", "a") do data
            println(data,
                "$(p.h)"
                * prod(" $(real(c2_n[2][i]))" for i in 1:3)
                # * prod(" $(real(c2_2_intra[i]))" for i in 1:3)
            )
        end
        # open("data_zeeman/$(p.savepath)_c2_3.txt", "a") do data
        #     println(data,
        #         "$(p.h)"
        #         * prod(" $(imag(c2_n[3][i]))" for i in 1:4)
        #     )
        # end
        open("data_zeeman/$(p.savepath)_c2_4.txt", "a") do data
            println(data,
                "$(p.h)"
                * prod(" $(real(c2_n[4][i]))" for i in 1:5)
            )
        end

        ##### test begin #####
        # open("data_zeeman/$(p.savepath)_test.txt", "a") do data
        #     println(data,
        #         "$(p.h) $(p.nk1) $(real(c2_n[2][1])) $(real(c2_2_intra[1])) $(real(c4_0)) $(real(c4_0_intra))"
        #     )
        # end
        ##### test end #####
    end

    nothing
end

main()
