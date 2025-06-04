function kpath_extract(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt
    open(p.Logstr, "a") do log
        println(log, "Now extract kpath and kmesh of GF, susceptibility...")
    end

    ### trace of G function
    giωk_tr::Array{Float64, 3} = sum(real.(@view(g.giωk[m.iω0_f, :, :, :, ζ, ζ]) for ζ in 1:p.nwan))
    k_HSP_path::Vector{Float64}, k_HSP::Vector{Float64}, k_HSP_tics::String, giωk_tr_HSP::Vector{ComplexF64} = kpath_extractor(
        p, sum(@view(g.giωk[m.iω0_f, :, :, :, ζ, ζ]) for ζ in 1:p.nwan)
    )

    ### largest eigenvalues of susceptibility
    χU_max::Array{Float64, 3} = Array{Float64}(undef, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3)
    @floop for (ik3, ik2, ik1) in my_iter((1:m.prmt.nk3, 1:m.prmt.nk2, 1:m.prmt.nk1))
        χU_max[ik1, ik2, ik3] = maximum(
            real, eigvals(@view(g.χ0iωk[m.iω0_b, ik1, ik2, ik3, :, :]) * m.U_mat)
        )
    end
    _, _, _, χU_max_HSP::Vector{Float64} = kpath_extractor(p, χU_max)

    ##### Save results
    h5open(p.plot_savepath, "cw") do file
        # kmap
        haskey(file, "kmap") && delete_object(file, "kmap")
        kmap = create_group(file, "kmap")
        kmap["χU_max"] = χU_max
        kmap["giωk_tr"] = giωk_tr

        # kpath
        haskey(file, "kpath") && delete_object(file, "kpath")
        kpath = create_group(file, "kpath")
        kpath["k_HSP_path"] = k_HSP_path
        kpath["k_HSP"] = k_HSP
        kpath["k_HSP_tics"] = k_HSP_tics
        kpath["χU_max_HSP"] = χU_max_HSP
        kpath["giωk_tr_HSP"] = giωk_tr_HSP
    end

    ### electric/magnetic susceptibility
    ops::Vector{Matrix{ComplexF64}} = [
        kron(σ0, σ0) ./ 2.0, # electric charge
        kron(σ3, σ0) ./ 2.0, # electric quadrupole
        kron(σ0, σ1) ./ 2.0, # uniform spin (x)
        kron(σ3, σ1) ./ 2.0, # staggered spin (x)
        kron(σ0, σ2) ./ 2.0, # uniform spin (y)
        kron(σ3, σ2) ./ 2.0, # staggered spin (y)
        kron(σ0, σ3) ./ 2.0, # uniform spin (z)
        kron(σ3, σ3) ./ 2.0  # staggered spin (z)
    ]
    suffixes::Vector{String} = ["ec", "eq", "uni_x", "sta_x", "uni_y", "sta_y", "uni_z", "sta_z"]

    for iop in eachindex(ops)
        χ_op::Array{Float64, 3} = real.(
            calc_multipole_susceptibility(m, g, ops[iop])[m.iω0_b, :, :, :]
        )
        _, _, _, χ_op_HSP::Vector{Float64} = kpath_extractor(p, χ_op)

        ##### Save results
        h5open(p.plot_savepath, "cw") do file
            write(file, "kmap/χ_$(suffixes[iop])", χ_op)
            write(file, "kpath/χ_$(suffixes[iop])_HSP", χ_op_HSP)
        end
    end

    open(p.Logstr, "a") do log
        println(log, "Done.")
    end

    k_HSP_path, k_HSP, χU_max_HSP, giωk_tr_HSP
end

function kpath_extract(m::Mesh, e::Eliashberg)
    p::Parameters = m.prmt
    open(p.Logstr, "a") do log
        println(log, "Now extract kpath and kmesh of gap function...")
    end

    ### gap function
    # sublattice basis
    Δ_sl::Array{ComplexF64, 5} = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
        Δ_sl[:, :, :, ζ1, ζ2] .= sum(
            @view(e.Δiωk[iω, :, :, :, ζ1, ζ2]) for iω in (m.iω0_f-1):m.iω0_f
        ) ./ 2.0
    end

    # band basis: U(k)^† Δ(k) U(-k)^*
    Δ_band::Array{ComplexF64, 5} = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        jk1::Int64 = mod(2-ik1, 1:p.nk1)
        jk2::Int64 = mod(2-ik2, 1:p.nk2)
        jk3::Int64 = mod(2-ik3, 1:p.nk3)
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            Δ_band[ik1, ik2, ik3, ζ1, ζ2] += (
                conj(m.uk[ik1, ik2, ik3, ζ3, ζ1])
                * sum(e.Δiωk[iω, ik1, ik2, ik3, ζ3, ζ4] for iω in (m.iω0_f-1):m.iω0_f) / 2.0
                * conj(m.uk[jk1, jk2, jk3, ζ4, ζ2])
            )
        end
    end

    ##### Save results
    h5open(p.plot_savepath, "cw") do file
        # sublattice-based SC order parameter
        name = "Δsl_$(p.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
        haskey(file, name) && delete_object(file, name)
        write(file, name, Δ_sl)

        # band-based SC order parameter
        name = "Δband_$(p.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
        haskey(file, name) && delete_object(file, name)
        write(file, name, Δ_band)
    end

    open(p.Logstr, "a") do log
        println(log, "Done.")
    end

    nothing
end

"Extracts points of given quantity along HSP k-path Γ->Y->M'->Γ->M->X->Γ."
function kpath_extractor(p::Parameters, quant::Array{T, 3}) where T
    ##### Path extraction
    k_HSP::Vector{Float64} = [0, 1, 2, 2+sqrt(2), 2+2*sqrt(2), 3+2*sqrt(2), 4+2*sqrt(2)] ./ (4+2*sqrt(2))
    k_HSP_tics::String = "set xtics ('{/Symbol G}' $(k_HSP[1]), 'Y' $(k_HSP[2]), 'M' $(k_HSP[3]), '{/Symbol G}' $(k_HSP[4]), 'M' $(k_HSP[5]), 'X' $(k_HSP[6]), '{/Symbol G}' $(k_HSP[end]))"

    ### Γ -> Y
    k_HSP_ΓY::Vector{Float64} = collect(range(k_HSP[1], k_HSP[2], length=p.nk2÷2+1))
    quant_HSP_ΓY::Vector{T} = quant[1, 1:p.nk2÷2+1, 1]

    ### Y -> M'
    k_HSP_YM2::Vector{Float64} = collect(range(k_HSP[2], k_HSP[3], length=p.nk1÷2+1))
    quant_HSP_YM2::Vector{T} = [
        quant[mod(p.nk1-it+2, 1:p.nk1), p.nk2÷2+1, 1] for it in 1:p.nk1÷2+1
    ]

    ### M' -> Γ
    k_HSP_M2Γ::Vector{Float64} = collect(range(k_HSP[3], k_HSP[4], length=p.nk1÷2+1))
    quant_HSP_M2Γ::Vector{T} = [
        quant[mod(p.nk1÷2+it, 1:p.nk1), p.nk2÷2-it+2, 1] for it in 1:p.nk1÷2+1
    ]

    ### Γ -> M
    k_HSP_ΓM::Vector{Float64} = collect(range(k_HSP[4], k_HSP[5], length=p.nk1÷2+1))
    quant_HSP_ΓM::Vector{T} = [
        quant[it, it, 1] for it in 1:p.nk1÷2+1
    ]

    ### M -> X
    k_HSP_MX::Vector{Float64} = collect(range(k_HSP[5], k_HSP[6], length=p.nk2÷2+1))
    quant_HSP_MX::Vector{T} = [
        quant[p.nk1÷2+1, p.nk2÷2-it+2, 1] for it in 1:p.nk2÷2+1
    ]

    ### X -> Γ
    k_HSP_XΓ::Vector{Float64} = collect(range(k_HSP[6], k_HSP[7], length=p.nk1÷2+1))
    quant_HSP_XΓ::Vector{T} = [
        quant[p.nk1÷2-it+2, 1, 1] for it in 1:p.nk1÷2+1
    ]

    # Extract along HSP k-line:
    k_HSP_path::Vector{Float64} = vcat(
        k_HSP_ΓY,
        k_HSP_YM2,
        k_HSP_M2Γ,
        k_HSP_ΓM,
        k_HSP_MX,
        k_HSP_XΓ
    )
    quant_HSP::Vector{T} = vcat(
        quant_HSP_ΓY,
        quant_HSP_YM2,
        quant_HSP_M2Γ,
        quant_HSP_ΓM,
        quant_HSP_MX,
        quant_HSP_XΓ
    )

    k_HSP_path, k_HSP, k_HSP_tics, quant_HSP
end

function calc_multipole_susceptibility(m::Mesh, g::Gfunction, op::Matrix{ComplexF64})
    p::Parameters = m.prmt

    χ_O = zeros(ComplexF64, m.bnω, p.nk1, p.nk2, p.nk3)
    for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
        ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
        ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
        for iq3 in 1:p.nk3, iq2 in 1:p.nk2, iq1 in 1:p.nk1
            qvec::Vector{Float64} = [
                2(iq1-1)%p.nk1 / p.nk1 - (2(iq1-1)÷p.nk1),
                2(iq2-1)%p.nk2 / p.nk2 - (2(iq2-1)÷p.nk2),
                2(iq3-1)%p.nk3 / p.nk3 - (2(iq3-1)÷p.nk3)
            ]

            # exp(iq.(r3-r1)): phase factor recovering the nonsymmorphic symmetry
            @views χ_O[:, iq1, iq2, iq3] .+= (
                (cispi(dot(qvec, p.pos[(ζ3-1)÷p.nspin+1] .- p.pos[(ζ1-1)÷p.nspin+1])) * op[ζ1, ζ2] * op[ζ4, ζ3])
                .* g.χiωk[:, iq1, iq2, iq3, ζ12, ζ34]
            )
        end
    end

    if maximum(abs, imag.(χ_O)) > 1e-10 
        open(p.Logstr, "a") do log
            println(log, "!!!!!! Imaginary part of susceptibility is very large: $(maximum(abs, imag.(χ_O))) !!!!!!")
        end
    end

    χ_O
end
