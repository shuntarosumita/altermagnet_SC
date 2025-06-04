##############################################################################
##### Main code: Calculate lin. eliashberg eq. within FLEX approximation #####
##############################################################################
mutable struct Eliashberg
    Viτr::Array{ComplexF64, 6}
    V_DC::Array{ComplexF64, 2}
    Δiωk::Array{ComplexF64, 6}
    fiτr::Array{ComplexF64, 6}
    fiτr_0::Array{ComplexF64, 2}
    iQ::Vector{Int64}
    λ::Float64
end

struct PointGroup
    grp::String
    elements::Vector{Int64}
    irrep::String
    character::Vector{Int64}
    sym_repres::Vector{Function}
    sym_id::Vector{Function}
end

function Eliashberg(m::Mesh, g::Gfunction, iQ::Vector{Int64})::Eliashberg
    p::Parameters = m.prmt

    Viτr = Array{ComplexF64, 6}(undef, m.bnτ, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    V_DC = Array{ComplexF64, 2}(undef, p.nwan^2, p.nwan^2)
    Δiωk = Array{ComplexF64, 6}(undef, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    fiτr = Array{ComplexF64, 6}(undef, m.fnτ, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    fiτr_0 = Array{ComplexF64, 2}(undef, p.nwan, p.nwan)

    e = Eliashberg(Viτr, V_DC, Δiωk, fiτr, fiτr_0, iQ, 0.0)

    set_Viτr!(m, g, e)

    return e
end

function save_Eliashberg(m::Mesh, g::Gfunction, e::Eliashberg)
    ##### Save results
    open(m.prmt.Logstr, "a") do log
        println(log, "Saving all data now...")
    end

    open(m.prmt.SC_EV_path, "a") do file
        println(file, "$(m.prmt.T) $(e.iQ[1]) $(e.iQ[2]) $(e.iQ[3]) $(e.λ)")
    end

    if e.iQ == [0, 0, 0]
        h5open(m.prmt.savepath, "cw") do file
            name = "eliashberg_$(m.prmt.SC_type)_$(e.iQ[1])_$(e.iQ[2])_$(e.iQ[3])"
            haskey(file, name) && delete_object(file, name)
            group = create_group(file, name)
            group["gap"] = e.Δiωk
            group["λ"] = e.λ
        end
    end
end

##############
### Self consistency loop for linearized Eliashberg equation
### Employs power iterative method to solve λ*Δ = λ*V*F in (τ, r)-space
##############
"""
Self consistency loop for super conduction parameter via eigenvalue method.
Implements FLEX approximation in linearized Eliashberg equation.
Handles depending on SC-type input in p.SC_type(=parameters) the equation differently.
"""
function solve_Eliashberg!(m::Mesh, g::Gfunction, e::Eliashberg)
    set_Δiωk!(m, g, e)
    e.λ = SC_sfc!(m, g, e, 0.0)

    ishift::Int = 1
    while e.λ < 0
        ishift > 10 && break

        open(m.prmt.Logstr, "a") do log
            println(log, "ishift = $ishift: Another eliashberg cycle will be performed.")
        end
        open(m.prmt.Logerrstr, "a") do logerr
            println(logerr, "$(m.prmt.err_str_begin) ishift = $ishift: λ < 0 => new round!")
        end

        open(m.prmt.SC_EV_path_neg, "a") do file
            println(file, "$(m.prmt.T) $(e.iQ[1]) $(e.iQ[2]) $(e.iQ[3]) $ishift $(e.λ)")
        end

        set_Δiωk!(m, g, e)
        e.λ = SC_sfc!(m, g, e, e.λ)

        # save_Eliashberg(m, g, e)
        ishift += 1
    end
end

function SC_sfc!(m::Mesh, g::Gfunction, e::Eliashberg, λ_in::Float64)
    p::Parameters = m.prmt
    nall::Int = m.fnω * p.nk * p.nwan^2

    λ::ComplexF64 = 0.0
    Δmax::Float64 = 10.0; λmax::Float64 = 10.0
    yiωk::Array{ComplexF64, 6} = zeros(ComplexF64, m.fnω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    pg::PointGroup = PointGroup(m)

    count::Int = 1
    while Δmax > 100*p.SC_sfc_tol || λmax > p.SC_sfc_tol || abs(imag(λ)) > p.SC_sfc_tol * abs(real(λ))
        # save previous λ and Δiωk
        λ_old::Float64 = real(λ)
        Δiωk_old::Array{ComplexF64, 6} = copy(e.Δiωk)

        # Power iteration method for computing λ
        set_fiτr!(m, g, e)

        # y = V*F
        yiτr = zeros(ComplexF64, m.fnτ, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
            ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
            yiτr[:, :, :, :, ζ1, ζ4] .+= @views(
                e.Viτr[:, :, :, :, ζ12, ζ34] .* e.fiτr[:, :, :, :, ζ2, ζ3]
            )
        end

        # Fourier transform
        yiτk::Array{ComplexF64, 6} = r_to_k(m, yiτr)
        yiωk .= τ_to_ωn(m, Fermionic(), yiτk)

        # y_HF
        for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
            ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
            ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
            yiωk[:, :, :, :, ζ1, ζ4] .+= e.V_DC[ζ12, ζ34] * e.fiτr_0[ζ2, ζ3] / p.nk
        end

        ### y - λ*y (power iteration method trick)
        yiωk .-= λ_in .* e.Δiωk

        ### Impose symmetry conditions
        symmetrize_gap!(m, e.iQ, pg, yiωk)

        ### Calculate λ
        λ = sum(conj(e.Δiωk) .* yiωk) / nall + λ_in
        λmax = abs(λ - λ_old)

        ### Calculate Δiωk
        e.Δiωk .= normalize(yiωk) .* sqrt(nall)
        Δmax = maximum(abs.(e.Δiωk .- Δiωk_old))

        # break a loop if λ converges to a negative value
        (real(λ) < 0.0 && λmax < 1e-4) && break

        (count > 1000) && break

        if count % 101 == 1
            open(p.Logstr, "a") do log
                println(log, "$count: $λ, $λmax, $Δmax")
            end
        end

        count += 1
    end

    open(p.Logstr, "a") do log
        println(log, "$count: $λ, $λmax, $Δmax; loop finished")
    end

    # finilize Δiωk
    maxΔid = findmax(abs, e.Δiωk)[2]
    e.Δiωk ./= e.Δiωk[maxΔid]

    real(λ)
end

function symmetrize_gap!(m::Mesh, iQ::Vector{Int64}, pg::PointGroup, yiωk::Array{ComplexF64, 6})
    # Even function of matsubara frequency
    # yiωk_inv corresponds to y(k, -iωn)
    yiωk_inv::Array{ComplexF64, 6} = reverse(yiωk, dims=1)
    yiωk .= (yiωk .+ yiωk_inv) ./ 2.0

    # Pauli exclusion priciple: y(k)_ab = -y(-k+Q)_ba
    # yiωk_inv corresponds to y(-k+Q, iωn)^T
    yiωk_inv .= reverse(
        circshift(yiωk, (0, -1, -1, -1, 0, 0)),
        dims=(2, 3, 4)
    )
    yiωk_inv .= permutedims(
        circshift(
            yiωk_inv,
            (0, iQ[1], iQ[2], iQ[3], 0, 0)
        ),
        (1, 2, 3, 4, 6, 5)
    )
    yiωk .= (yiωk .- yiωk_inv) ./ 2.0

    # point group symmetry (not applicable to FFLO)
    iQ == [0, 0, 0] && (yiωk .= projection_operator(m, pg, yiωk))

    yiωk
end

### Set Coulomb interaction V_a(τ, r) --------------------------------
function set_Viτr!(m::Mesh, g::Gfunction, e::Eliashberg)
    # Set V
    Viωk = Array{ComplexF64, 6}(undef, m.bnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2, m.prmt.nwan^2)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.bnω
        Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
            -m.U_mat * g.χiωk[iω, ik1, ik2, ik3, :, :] * m.U_mat
        )
    end
    e.V_DC .= -m.U_mat ./ 2.0

    # Fourier transform
    Viωr::Array{ComplexF64, 6} = k_to_r(m, Viωk)
    e.Viτr .= ωn_to_τ(m, Bosonic(), Viωr)
end

### Set inital gap Δ0(iω_n, k) --------------------------------------
"""
Set initial guess for gap function according to system symmetry.
The setup is carried out in real space and then FT.
"""
function set_Δiωk!(m::Mesh, g::Gfunction, e::Eliashberg)
    p::Parameters = m.prmt

    ### set basis functions
    Δpx = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δpy = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δdxy = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    Δdx2y2 = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    ΔfA1 = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    ΔfB2 = Array{ComplexF64, 3}(undef, p.nk1, p.nk2, p.nk3)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::Float64 = (2π*(ik1-e.iQ[1]/2-1)) / p.nk1
        k2::Float64 = (2π*(ik2-e.iQ[2]/2-1)) / p.nk2
        Δpx[ik1, ik2, ik3] = sin(k1)
        Δpy[ik1, ik2, ik3] = sin(k2)
        Δdx2y2[ik1, ik2, ik3] = cos(k1) - cos(k2)
        Δdxy[ik1, ik2, ik3] = sin(k1) * sin(k2)
        ΔfA1[ik1, ik2, ik3] = 1.0 + cis(k1) + cis(k1-k2) + cis(-k2)
        ΔfB2[ik1, ik2, ik3] = 1.0 - cis(k1) + cis(k1-k2) - cis(-k2)
    end

    ### Set inital gap function according to symmetry
    δ::Float64 = 0.0
    iσ2 = 1.0im .* σ2
    if p.SC_type == "A1"
        e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
            .+ ones(ComplexF64, p.nk1, p.nk2, p.nk3) ⊗ kron(σ0, iσ2) # singlet: s τ0
            .+ δ .* real.(ΔfA1) ⊗ kron(σ1, iσ2) .- imag.(ΔfA1) ⊗ kron(σ2, iσ2) # singlet: extended-s τ{x,y}
            .+ δ .* Δpx ⊗ kron(σ0, σ2 * iσ2) # triplet: + px τ0 sy
            .- δ .* Δpy ⊗ kron(σ0, σ1 * iσ2) # triplet: - py τ0 sx
        )
    elseif p.SC_type == "A2"
        e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
            Δdx2y2 ⊗ kron(σ3, iσ2) # singlet: dx2y2 τz
            .+ δ .* Δpx ⊗ kron(σ0, σ1 * iσ2) # triplet: + px τ0 sx
            .+ δ .* Δpy ⊗ kron(σ0, σ2 * iσ2) # triplet: + py τ0 sy
        )
    elseif p.SC_type == "B1"
        e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
            Δdx2y2 ⊗ kron(σ0, iσ2) # singlet: dx2y2 τ0
            .+ δ .* Δpx ⊗ kron(σ0, σ2 * iσ2) # triplet: + px τ0 sy
            .+ δ .* Δpy ⊗ kron(σ0, σ1 * iσ2) # triplet: + py τ0 sx
        )
    elseif p.SC_type == "B2"
        e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ (
            real.(ΔfB2) ⊗ kron(σ1, iσ2) .- imag.(ΔfB2) ⊗ kron(σ2, iσ2) # singlet: dxy τ{x,y}
            .+ δ .* Δdxy ⊗ kron(σ0, iσ2) # singlet: dxy τ0
            .+ δ .* Δpx ⊗ kron(σ0, σ1 * iσ2) # triplet: + px τ0 sx
            .- δ .* Δpy ⊗ kron(σ0, σ2 * iσ2) # triplet: - py τ0 sy
        )
    end

    e.Δiωk .*= sqrt(m.fnω * p.nk * p.nwan^2) / norm(e.Δiωk)

    e.Δiωk
end

# function set_Δiωk!(m::Mesh, g::Gfunction, e::Eliashberg)
#     p::Parameters = m.prmt

#     ### Set inital gap function according to symmetry
#     e.Δiωk .= ones(ComplexF64, m.fnω) ⊗ rand(ComplexF64, p.nk1, p.nk2, p.nk3)
#     e.Δiωk .= projection_operator(m, PointGroup(m), e.Δiωk)

#     # Pauli exclusion priciple: y(k)_ab = -y(-k+Q)_ba
#     # yiωk_inv corresponds to y(-k+Q, iωn)^T
#     Δiωk_inv::Array{ComplexF64, 6} = reverse(
#         circshift(e.Δiωk, (0, -1, -1, -1, 0, 0)),
#         dims=(2, 3, 4)
#     )
#     Δiωk_inv .= permutedims(
#         circshift(
#             Δiωk_inv,
#             (0, e.iQ[1], e.iQ[2], e.iQ[3], 0, 0)
#         ),
#         (1, 2, 3, 4, 6, 5)
#     )
#     e.Δiωk .= (e.Δiωk .- Δiωk_inv) ./ 2.0

#     e.Δiωk .*= sqrt(m.fnω * p.nk * p.nwan^2) / norm(e.Δiωk)

#     e.Δiωk
# end

### Set anomalous Green function F(τ, r) --------------------------------------
function set_fiτr!(m::Mesh, g::Gfunction, e::Eliashberg)
    # G(-k, -iωn)
    giωk_invk::Array{ComplexF64, 6} = reverse(
        circshift(g.giωk, (0, -1, -1, -1, 0, 0)),
        dims=(1, 2, 3, 4)
    )
    # G(-k+Q, -iωn)^T
    giωk_invk = permutedims(
        circshift(
            giωk_invk,
            (0, e.iQ[1], e.iQ[2], e.iQ[3], 0, 0)
        ),
        (1, 2, 3, 4, 6, 5)
    )

    fiωk = Array{ComplexF64, 6}(undef, m.fnω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan, m.prmt.nwan)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.fnω
        fiωk[iω, ik1, ik2, ik3, :, :] .= @views(
            -g.giωk[iω, ik1, ik2, ik3, :, :] * e.Δiωk[iω, ik1, ik2, ik3, :, :] * giωk_invk[iω, ik1, ik2, ik3, :, :]
        )
    end

    # Fourier transform
    fiωr::Array{ComplexF64, 6} = k_to_r(m, fiωk)
    e.fiτr .= ωn_to_τ(m, Fermionic(), fiωr)

    f_l::Array{ComplexF64, 3} = fit(m.IR_basis_set.smpl_tau_f, e.fiτr[:, 1, 1, 1, :, :], dim=1)
    for ζ2 in 1:m.prmt.nwan, ζ1 in 1:m.prmt.nwan
        e.fiτr_0[ζ1, ζ2] = dot(m.IR_basis_set.basis_f.u(0), @view(f_l[:, ζ1, ζ2]))
    end
end


##### Symmetry arguments (not used for FFLO state) #####
minus_k(ik::Int, nk::Int) = mod(2-ik, 1:nk)

function PointGroup(m::Mesh)
    p::Parameters = m.prmt

    # determine point group
    if p.h == 0.0
        grp = "C4v"
        elements = [1, 2, 3, 4, 5, 6, 7, 8]
    else
        if p.h_dir == "001"
            grp = "C2v"
            elements = [1, 2, 5, 6]
        elseif p.h_dir == "100"
            grp = "Cs"
            elements = [1, 6]
        elseif p.h_dir == "010"
            grp = "Cs"
            elements = [1, 5]
        elseif p.h_dir == "110"
            grp = "Cs"
            elements = [1, 8]
        end
    end

    # character (E, C2z, C4z, C4z^3, Mx, My, M11, M1-1)
    irrep::String = p.SC_type
    character::Vector{Int64} = (
        if irrep == "A1"
            [1, 1, 1, 1, 1, 1, 1, 1]
        elseif irrep == "A2"
            [1, 1, 1, 1, -1, -1, -1, -1]
        elseif irrep == "B1"
            [1, 1, -1, -1, 1, 1, -1, -1]
        elseif irrep == "B2"
            [1, 1, -1, -1, -1, -1, 1, 1]
        end
    )

    # symmetry representations
    sym_repres::Vector{Function} = [
        (k1, k2, k3) -> Matrix{ComplexF64}(I, p.nwan, p.nwan), # E
        (k1, k2, k3) -> kron([cis(-k2) 0; 0 cis(-k1)], 1.0im .* σ3), # C2z
        (k1, k2, k3) -> kron([0 cis(-k2); 1 0], exp(1im*π/4 .* σ3)), # C4z
        (k1, k2, k3) -> kron([0 1; cis(-k1) 0], exp(3im*π/4 .* σ3)), # C4z^3
        (k1, k2, k3) -> (kron([0 cis(-k2); 1 0], 1.0im .* σ1)), # Mx
        (k1, k2, k3) -> (kron([0 1; cis(-k1) 0], 1.0im .* σ2)), # My
        (k1, k2, k3) -> (kron([cis(-k2) 0; 0 cis(-k1)], 1.0im .* exp(1im*π/4 .* σ3) * σ1)), # M11
        (k1, k2, k3) -> (kron(σ0, 1.0im .* exp(3im*π/4 .* σ3) * σ1)) # M1-1
    ]

    # indices transformed by symmetry operator
    ## remark: considering the inverse (k -> p^{-1} k)
    sym_id::Vector{Function} = [
        (ik1, ik2, ik3) -> [ik1, ik2, ik3], # E
        (ik1, ik2, ik3) -> [minus_k(ik1, p.nk1), minus_k(ik2, p.nk2), ik3], # C2z
        (ik1, ik2, ik3) -> [ik2, minus_k(ik1, p.nk2), ik3], # C4z
        (ik1, ik2, ik3) -> [minus_k(ik2, p.nk1), ik1, ik3], # C4z^3
        (ik1, ik2, ik3) -> [minus_k(ik1, p.nk1), ik2, ik3], # Mx
        (ik1, ik2, ik3) -> [ik1, minus_k(ik2, p.nk2), ik3], # My
        (ik1, ik2, ik3) -> [ik2, ik1, ik3], # M11
        (ik1, ik2, ik3) -> [minus_k(ik2, p.nk1), minus_k(ik1, p.nk2), ik3] # M1-1
    ]

    PointGroup(grp, elements, irrep, character, sym_repres, sym_id)
end

"""
Point group projection operator for order parameter
"""
function projection_operator(m::Mesh, pg::PointGroup, yiωk::Array{ComplexF64})
    p::Parameters = m.prmt

    yiωk_new::Array{ComplexF64} = zeros(ComplexF64, size(yiωk)...)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1, ip in pg.elements
        # indices transformed by symmetry operator
        jk1::Int64, jk2::Int64, jk3::Int64 = pg.sym_id[ip](ik1, ik2, ik3)
        gk1::Float64 = (2π*(jk1-1)) / p.nk1
        gk2::Float64 = (2π*(jk2-1)) / p.nk2
        gk3::Float64 = (2π*(jk3-1)) / p.nk3

        # act projection operator
        for iω in 1:m.fnω
            yiωk_new[iω, ik1, ik2, ik3, :, :] .+= pg.character[ip] .* @views(
                pg.sym_repres[ip](gk1, gk2, gk3)' * yiωk[iω, jk1, jk2, jk3, :, :] * conj.(pg.sym_repres[ip](-gk1, -gk2, -gk3))
            )
        end
    end
    yiωk_new .*= pg.character[1] ./ length(pg.elements)

    yiωk_new
end
