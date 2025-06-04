##############
# Set parameters for current calculation
##############
mutable struct Parameters
    mode::String
    mix::Float64
    round_it::Int64

    SC_type::String

    IR_tol::Float64
    g_sfc_tol::Float64
    SC_sfc_tol::Float64

    system::String
    nk1::Int64
    nk2::Int64
    nk3::Int64
    nk::Int64
    T::Float64
    β::Float64
    ωmax::Float64
    n_fill::Float64
    nspin::Int64
    norb::Int64
    nwan::Int64
    U::Float64
    h::Float64
    h_dir::String
    α::Float64
    pos::Vector{Vector{Float64}}

    Logstr::String
    Logerrstr::String
    err_str_begin::String
    savepath::String
    loadpath::String
    plot_savepath::String
    plot_loadpath::String
    data_dir::String
    
    BSE_EV_path::String
    SC_EV_path::String
    SC_EV_path_neg::String
end

function Parameters(
        system::String,
        SC_type::String,
        h::Float64,
        h_dir::String,
        α::Float64,
        n::Float64,
        U::Float64,
        T::Float64,
        round_it::Int64,
        mode::String;
        nk1::Int64,
        nk2::Int64,
        nk3::Int64,
        ωmax::Float64,
        h_load::Float64=h,
        α_load::Float64=α,
        U_load::Float64=U,
        data_dir::String="./"
    )::Parameters

    # General settings
    mix::Float64 = 0.2 # Value of how much of the new G is to be used

    # Cutoffs/accuracy
    IR_tol::Float64 = 1e-15
    g_sfc_tol::Float64 = 1e-6
    SC_sfc_tol::Float64 = 1e-4

    # Physical quantities
    nk::Int64 = nk1 * nk2 * nk3
    β::Float64 = 1/T
    n_fill::Float64 = n
    nspin::Int64 = 2
    norb::Int64 = 2
    nwan::Int64 = nspin * norb

    # coordinate of the molecules inside the unit cell
    pos::Vector{Vector{Float64}} = [
        [0.0, -0.5, 0.0], # sublattice 1
        [-0.5, 0.0, 0.0]  # sublattice 2
    ]

    ### Log options
    Log_name::String = Printf.format(
        Printf.Format("log_$(system)/Log_h$(h_dir)_%.3f_a_%.3f_n_%.3f_U_%.3f_T_%.4f"),
        h, α, n_fill, U, T
    )
    Logstr::String = Log_name * ".txt"
    Logerrstr::String = Log_name * "_err.txt"
    err_str_begin::String = @sprintf(
        "System h%s = %.3f | α = %.3f | n = %.3f | U = %.3f | T = %.4f :",
        h_dir, h, α, n_fill, U, T
    )

    ### Setting saving options
    calc_name::String = "$(data_dir)/Odata_$(system)/$(mode)/data_h$(h_dir)_%.3f_a_%.3f_n_%.3f_U_%.3f_T_%.4f.h5"
    plot_name::String = "$(data_dir)/Odata_$(system)/$(mode)/plot_data_h$(h_dir)_%.3f_a_%.3f_n_%.3f_U_%.3f_T_%.4f.h5"

    # formatting middle string
    savepath::String = Printf.format(Printf.Format(calc_name), h, α, n_fill, U, T)
    loadpath::String = Printf.format(Printf.Format(calc_name), h_load, α_load, n_fill, U_load, T)
    plot_savepath::String = Printf.format(Printf.Format(plot_name), h, α, n_fill, U, T)
    plot_loadpath::String = Printf.format(Printf.Format(plot_name), h_load, α_load, n_fill, U_load, T)

    # eigenvalue strings
    BSE_EV_path::String = Printf.format(
        Printf.Format("BSE_kernel_EV_$(system)/$(mode)/max_ev_n_%.3f_U_%.3f.txt"),
        n_fill, U
    )
    SC_EV_path::String = Printf.format(
        Printf.Format("SC_EV_$(system)/$(mode)/$(SC_type)_lam_h$(h_dir)_%.3f_a_%.3f_n_%.3f_U_%.3f.txt"),
        h, α, n_fill, U
    )
    SC_EV_path_neg::String = Printf.format(
        Printf.Format("SC_EV_$(system)/$(mode)/$(SC_type)_lam_h$(h_dir)_%.3f_a_%.3f_n_%.3f_U_%.3f_negative.txt"),
        h, α, n_fill, U
    )

    return Parameters(
        mode, mix, round_it,
        SC_type,
        IR_tol, g_sfc_tol, SC_sfc_tol,
        system, nk1, nk2, nk3, nk, T, β, ωmax, n_fill, nspin, norb, nwan, U, h, h_dir, α, pos,
        Logstr, Logerrstr, err_str_begin, savepath, loadpath, plot_savepath, plot_loadpath, data_dir,
        BSE_EV_path, SC_EV_path, SC_EV_path_neg
    )
end

function save_Parameters(p::Parameters)
    ### Generate directories/h5 file if not exist
    isdir("log_$(p.system)") || mkdir("log_$(p.system)")
    isdir("SC_EV_$(p.system)") || mkdir("SC_EV_$(p.system)")
    isdir("SC_EV_$(p.system)/$(p.mode)") || mkdir("SC_EV_$(p.system)/$(p.mode)")
    isdir("BSE_kernel_EV_$(p.system)") || mkdir("BSE_kernel_EV_$(p.system)")
    isdir("BSE_kernel_EV_$(p.system)/$(p.mode)") || mkdir("BSE_kernel_EV_$(p.system)/$(p.mode)")
    isdir("$(p.data_dir)/Odata_$(p.system)") || mkdir("$(p.data_dir)/Odata_$(p.system)")
    isdir("$(p.data_dir)/Odata_$(p.system)/$(p.mode)") || mkdir("$(p.data_dir)/Odata_$(p.system)/$(p.mode)")

    isfile(p.savepath) || (
        h5open(p.savepath, "w") do file
            write(file, "SystemName", "$(p.system)")
            write(file, "N_k1", p.nk1)
            write(file, "N_k2", p.nk2)
            write(file, "N_k3", p.nk3)
            write(file, "IR_ωmax", p.ωmax)
            write(file, "IR_tol", p.IR_tol)
            write(file, "g_sfc_tol", p.g_sfc_tol)
            write(file, "SC_sfc_tol", p.SC_sfc_tol)
            write(file, "n_fill", p.n_fill)
            write(file, "T", p.T)
            write(file, "U", p.U)
            write(file, "h", p.h)
            write(file, "h_dir", p.h_dir)
            write(file, "α", p.α)
        end
    )
end
