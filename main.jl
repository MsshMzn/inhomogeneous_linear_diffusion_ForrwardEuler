using Calculus
using CSV
using LinearAlgebra
using DataFrames
using Plots

# --- ディレクトリの作成 ---
#mkpath("./csv")
#mkpath("./png")

# --- 物理モデル・初期条件の定義 ---
const mu_0_center = 1.0
const mu_0_coeff = 0.5
const mu_0_osc = 5.0 * pi

function mu_0(x)
    return mu_0_center + mu_0_coeff * cos(mu_0_osc * x)
end

const λ = 1.0
function phi(x)
    return (0.5 * λ) * x^2
end

d_center = 1.0
d_coeff = 0.5
d_osc = 4.0 * pi

function d(x)
    return d_center + d_coeff*sin(d_osc*x)
end

# --- 数値計算パラメータ ---
const k = 0.25
const N = 100
const T = 10.0

const dx = 1.0 / N
const dt = k * dx^2 / 1.5
const Iterate = floor(Int, T / dt)
const mod_time = 6

# --- 行列構築関数 ---
function build_Inhom_Neumann_Laplace(dim::Int, vec, width)
    D = zeros(dim + 1, dim + 1)
    w2 = width^2

    D[1, 1] = -2.0 * vec[1] / w2
    D[1, 2] = 2.0 * vec[1] / w2

    for i in 2:dim
        D[i, i]   = -2.0 * vec[i] / w2
        D[i, i-1] = 1.0 * vec[i] / w2
        D[i, i+1] = 1.0 * vec[i] / w2
    end

    D[dim+1, dim]   = 2.0 * vec[dim+1] / w2
    D[dim+1, dim+1] = -2.0 * vec[dim+1] / w2

    return D
end


# --- 物理量の計算マシナリー ---
function trapezoid_integral(x_SpaceVariable, y_RangeVariable)
    VectorLength = length(x_SpaceVariable)
    trapezoid_integral_vector = ones(VectorLength)
    trapezoid_integral_vector[1] = 0.5
    trapezoid_integral_vector[VectorLength] = 0.5
    h = x_SpaceVariable[2] - x_SpaceVariable[1]
    return h * dot(y_RangeVariable, trapezoid_integral_vector)
end

function density(unk::Vector{Float64}, phi_dis, d_dis)
    return exp.((unk .- phi_dis) ./ d_dis)
end

function list_derivative_zero_Neumann(unk::Vector{Float64}, width::Float64)
    dim = length(unk)
    D = zeros(dim, dim)
    dx = 2*width
    for i in 2:dim-1
        D[i, i+1] = 1.0/ dx
        D[i, i-1] = -1.0 / dx
    end
    return D * unk
end

function external_force(unk, width::Float64, phi_dis, phi_derivative_dis, log_d_derivative_dis)
    unk_derivative = list_derivative_zero_Neumann(unk, width)
    return (-(unk .- phi_dis) .* log_d_derivative_dis .+ unk_derivative .- phi_derivative_dis) .* unk_derivative 
end

function next_step(unk::Vector{Float64}, space_width::Float64, time_width::Float64, Df, phi_dis, phi_derivative_dis, log_d_derivative_dis)
    return Df * unk .+ time_width .* external_force(unk, space_width, phi_dis, phi_derivative_dis, log_d_derivative_dis)
end

function dissipation(x_SpaceVariable, unk::Vector{Float64}, phi_dis, d_dis)
    width = x_SpaceVariable[2] - x_SpaceVariable[1]
    unk_derivative = list_derivative_zero_Neumann(unk, width)
    density_dis = density(unk, phi_dis, d_dis)
    return trapezoid_integral(x_SpaceVariable, (unk_derivative.^2) .* density_dis)
end

function energy(x_SpaceVariable, unk::Vector{Float64}, phi_dis, d_dis)
    dens = density(unk, phi_dis, d_dis)
    # 配列生成を排除して計算
    energy_density = (d_dis .* (log.(dens) .- 1.0) .+ phi_dis) .* dens 
    return trapezoid_integral(x_SpaceVariable, energy_density)
end

# --- メイン実行ルーチン ---
function run_simulation()
    # 空間離散化
    mu_0_dis = zeros(N + 1)
    phi_dis = zeros(N + 1)
    phi_derivative_dis = zeros(N + 1)
    d_dis = zeros(N + 1)
    log_d_derivative_dis = zeros(N + 1)

    d_derivative = derivative(d)

    for i in 1:(N + 1)
        xi = dx * (i - 1)
        mu_0_dis[i] = mu_0(xi)
        phi_dis[i] = phi(xi)
        phi_derivative_dis[i] = derivative(phi, xi)
        d_dis[i] = d(xi)
        log_d_derivative_dis[i] = d_derivative(xi) / d(xi)
    end

    D = build_Inhom_Neumann_Laplace(N, d_dis, dx)
    Df = I + dt * D

    mu = copy(mu_0_dis)
    x_vals = LinRange(0.0, 1.0, N + 1)

    # 必要な保存サイズのみメモリを確保（deleteat! を撲滅）
    save_len = div(Iterate, mod_time) + 1
    RealTime = zeros(save_len)
    FreeEnergy_RealTime = zeros(save_len)
    Dissipation_RealTime = zeros(save_len)
    Dissipation_SpaceRealTime = zeros(save_len)

    # 初期状態
    RealTime[1] = 0.0
    FreeEnergy_RealTime[1] = energy(x_vals, mu, phi_dis, d_dis)
    Dissipation_RealTime[1] = dissipation(x_vals, mu, phi_dis, d_dis)
    Dissipation_SpaceRealTime[1] = 0.0

    println("Loop started. Total Iterations: ", Iterate)

    # タイムを発展させるメインループ
    for step in 1:Iterate
        mu = next_step(mu, dx, dt, Df, phi_dis, phi_derivative_dis, log_d_derivative_dis)

        if step % mod_time == 0
            idx = div(step, mod_time) + 1
            if idx <= save_len
                RealTime[idx] = step * dt * mod_time  # 元のロジックに追随
                FreeEnergy_RealTime[idx] = energy(x_vals, mu, phi_dis, d_dis)
                Dissipation_RealTime[idx] = dissipation(x_vals, mu, phi_dis, d_dis)
                Dissipation_SpaceRealTime[idx] = Dissipation_SpaceRealTime[idx - 1] + mod_time * dt * Dissipation_RealTime[idx]
            end
            if FreeEnergy_RealTime[idx] > FreeEnergy_RealTime[idx-1]
                break
                println("FreeEnergy is increasing : stopped.")
                open("diff-inhom_rate.txt", "w") do out
                println(out, "FreeEnergy is increasing : stopped.")
                end
            end
        end
    end

    # データ出力
    df = DataFrame(
        time = RealTime, 
        FreeEnergy = FreeEnergy_RealTime, 
        Dissipation = Dissipation_RealTime, 
        Dissipation_SpaceTime = Dissipation_SpaceRealTime, 
        Dissipation_Log10 = log10.(abs.(Dissipation_RealTime) .+ 1e-15)
    )
    CSV.write("./diff-free_energy.csv", df)

    # エネルギーグラフ
    plot(RealTime, FreeEnergy_RealTime, xlims=(0, 10), label="Free Energy(t)", xlabel="time")
    title!("d(x)=$d_center + $d_coeff sin ($d_osc x)")
    savefig("./diff-FreeEnergy.png")

    # レート解析
    k_val = maximum(abs.(log_d_derivative_dis))
    k_sqrt = maximum(abs.(log_d_derivative_dis .* sqrt.(d_dis)))

    rate_list = fill(10000.0, length(RealTime))
    for num in 1:(length(RealTime) - 1)
        if Dissipation_RealTime[num] > 1e-10
            rate_list[num] = (log10(Dissipation_RealTime[num]) - log10(Dissipation_RealTime[num + 1] + 1e-15)) / (mod_time * dt)
        end
    end
    rate = minimum(rate_list)

    open("diff-inhom_rate.txt", "w") do out
        println(out, "mu_0(x)=$mu_0_center + $mu_0_coeff cos ($mu_0_osc x)")
        println(out, "phi(x) = ($λ/2)*x^2")
        println(out, "d(x)=$d_center + $d_coeff sin ($d_osc x)")
        println(out, "Abs(derivative of log(d))=$k_val")
        println(out, "Abs(derivative of log(d) times sqrt(d))=$k_sqrt")
        println(out, "Dissipation_log10_Rate=$rate")
    end

    # 散逸グラフ
    plot(RealTime, log10.(abs.(Dissipation_RealTime) .+ 1e-15), color=:blue, xlims=(0, 10), label="Log(Dissipation(t))", xlabel="time")
    title!("d(x)=$d_center + $d_coeff sin ($d_osc x)")
    savefig("./diff-scale2.png")

    # 解プロファイル
    plot(x_vals, mu, color=:blue, xlims=(0, 1), ylims=(-2, 2), label="μ(x,10)", xlabel="x") 
    title!("d(x)=$d_center + $d_coeff sin ($d_osc x)")
    savefig("./diff-mu_solution_split100.png")

    println("Finished. Rate: ", rate)
end


run_simulation()