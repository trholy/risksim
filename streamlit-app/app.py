import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from risksim.copula import CopulaSimulator, CopulaConfig
from risksim.experiment import RiskExperiment, RiskExperimentConfig
from risksim.utils import StockDataLoader, PortfolioAnalyzer
from risksim.risk import RiskMeasure

st.set_page_config(page_title="RiskSim", layout="wide")

st.title("📈 RiskSim – Comprehensive Risk Simulation Framework")

st.sidebar.header("Simulation Parameters")

# --- Copula configuration inputs ---
n_runs_cfg = st.sidebar.number_input(
    "Number of Monte Carlo runs",
    min_value=100, max_value=100000,
    value=10000, step=1000)

st.sidebar.subheader("X Range")
x_min_cfg = st.sidebar.number_input(
    "X min", min_value=1, max_value=100, value=10, step=1)
x_max_cfg = st.sidebar.number_input(
    "X max", min_value=1, max_value=100, value=20, step=1)

st.sidebar.subheader("Y Range")
y_min_cfg = st.sidebar.number_input(
    "Y min", min_value=1, max_value=100, value=8, step=1)
y_max_cfg = st.sidebar.number_input(
    "Y max", min_value=1, max_value=100, value=22, step=1)

st.sidebar.subheader("Normal Distribution Parameters")
mu1_cfg = st.sidebar.number_input(
    "μ₁ (mean of X)", min_value=1, max_value=100, value=2, step=1)
mu2_cfg = st.sidebar.number_input(
    "μ₂ (mean of Y)", min_value=1, max_value=100, value=3, step=1)
std1_cfg = st.sidebar.number_input(
    "σ₁ (std of X)", min_value=1, max_value=100, value=2, step=1)
std2_cfg = st.sidebar.number_input(
    "σ₂ (std of Y)", min_value=1, max_value=100, value=3, step=1)
corr_cfg = st.sidebar.slider(
    "Correlation ρ",
    min_value=-0.99, max_value=0.99,
    value=0.0, step=0.05)

copula_cfg = CopulaConfig(
    n_runs=n_runs_cfg,
    x_range=(x_min_cfg, x_max_cfg),
    y_range=(y_min_cfg, y_max_cfg),
    mu=[mu1_cfg, mu2_cfg],
    std=[std1_cfg, std2_cfg],
    corr=[corr_cfg])

st.sidebar.header("Risk Experiment Parameters")
runs_cfg = st.sidebar.number_input(
    "Number of experiments", min_value=1, max_value=200, value=10)
sim_samples_cfg = st.sidebar.number_input(
    "Samples per experiment", min_value=50, max_value=5000, value=100)
alpha_cfg = st.sidebar.number_input(
    "Alpha (Quantile / VaR)", min_value=0.05, max_value=0.5, value=0.05)
gamma_cfg = st.sidebar.number_input(
    "Gamma (for Power Spectral Risk Measure)", min_value=0.1, max_value=1.0, value=0.5)

st.sidebar.header("Historical Data Settings")
ticker1 = st.sidebar.text_input("Ticker 1", value="VOW3.DE")
ticker2 = st.sidebar.text_input("Ticker 2", value="FME.DE")
data_dir = st.sidebar.text_input("Data Directory", value="../datasets")

exp_cfg = RiskExperimentConfig(
    runs=runs_cfg, sim_samples=sim_samples_cfg, alpha=alpha_cfg, gamma=gamma_cfg)

st.markdown("---")

# --- Run Simulation ---
st.subheader("🔹 Copula Simulation")

if st.button("Run Simulation"):
    sim = CopulaSimulator(copula_cfg)
    results = sim.run(full_log=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Histogram of X + Y")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(results["sum"], bins=50,
                density=True, alpha=0.7, edgecolor="black")
        ax.set_title("Distribution of X+Y (Gaussian Copula)")
        ax.set_xlabel("Sum")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    with col2:
        st.write("### Scatter: Correlated Normals (Z₁, Z₂)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            results["standard_norm_ab"][:, 0],
            results["standard_norm_ab"][:, 1], s=10, alpha=0.6)
        ax.set_xlabel("Z₁")
        ax.set_ylabel("Z₂")
        ax.set_title("Correlated Standard Normals")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    st.write("### Scatter: Gaussian Copula Realisations (U₁, U₂)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        results["copula_realisation"][:, 0],
        results["copula_realisation"][:, 1], s=10, alpha=0.6)
    ax.set_xlabel("U₁")
    ax.set_ylabel("U₂")
    ax.set_title("Realisations of Gauss-Copula")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.write("### Scatter: Transformed values of X, Y (Uniform Marginals)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(results["xy"][:, 0], results["xy"][:, 1], s=10, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Uniform Marginals X, Y with Gauss-Copula")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

st.markdown("---")

# --- Run Risk Experiment ---
st.subheader("🔹 Portfolio Risk Experiment")

if st.button("Run Experiment"):
    experiment = RiskExperiment(
        copula_config=copula_cfg, experiment_config=exp_cfg)
    df = experiment.run()

    st.write("### Risk Measure Results")
    st.dataframe(df.round(3))

    st.write("### Summary Statistics")
    min_max = {
        "VaR": (df["VaR"].min(), df["VaR"].max()),
        "CVaR": (df["CVaR"].min(), df["CVaR"].max()),
        "Power": (df["Power"].min(), df["Power"].max())}

    for key, (vmin, vmax) in min_max.items():
        delta = (vmin / vmax - 1) * 100
        st.markdown(f"**{key}:** min = {vmin:.2f}, max = {vmax:.2f}, Δ = {delta:.2f}%")

    # Plot cumulative histograms of all runs
    st.write("### Cumulative Distributions Across Runs")
    all_runs = df.attrs["all_sums"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, run_values in enumerate(all_runs):
        H, X1 = np.histogram(run_values, bins=len(run_values), density=True)
        dx = X1[1] - X1[0]
        F1 = np.cumsum(H) * dx
        ax.plot(X1[1:], F1, label=f"Run {idx + 1}", alpha=0.7)
    ax.set_title("Cumulative Distribution Function")
    ax.set_xlabel("Portfolio sum (X + Y)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    st.pyplot(fig)

# --- Historical Portfolio Analysis ---
st.markdown("---")
st.subheader("🔹 Historical Portfolio Analysis (Real Data)")

if st.button("Run Historical Analysis"):

    tickers = [ticker1, ticker2]

    # Step 1: Load data
    loader = StockDataLoader(tickers=tickers, data_dir=data_dir)
    data = loader.load_data()

    # Step 2: Compute returns & portfolio stats
    analyzer = PortfolioAnalyzer(data)
    v1 = analyzer.get_returns(tickers[0])
    v2 = analyzer.get_returns(tickers[1])
    corr_cfg = analyzer.calculate_correlation(tickers[0], tickers[1])
    mu1_cfg, std1_cfg = analyzer.calculate_statistics(v1)
    mu2_cfg, std2_cfg = analyzer.calculate_statistics(v2)

    v_pf = analyzer.get_portfolio_returns() # Historical (empirical) returns
    mu_pf, std_pf = analyzer.calculate_statistics(v_pf)

    # These are your key series:
    n_samples = len(v_pf)
    var_covar_results = np.random.normal(mu_pf, std_pf, size=n_samples)

    # Step 3: Display results
    st.write("### Summary Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label=f"{tickers[0]} Mean Return", value=f"{mu1_cfg:.4f}")
        st.metric(label=f"{tickers[0]} Std Dev", value=f"{std1_cfg:.4f}")
    with col2:
        st.metric(label=f"{tickers[1]} Mean Return", value=f"{mu2_cfg:.4f}")
        st.metric(label=f"{tickers[1]} Std Dev", value=f"{std2_cfg:.4f}")

    st.info(f"Correlation between {tickers[0]} and {tickers[1]}: **{corr_cfg:.4f}**")

    st.write("### Portfolio Performance")
    st.metric(label="Portfolio Mean Return", value=f"{mu_pf:.4f}")
    st.metric(label="Portfolio Std Dev", value=f"{std_pf:.4f}")

    # Historical (empirical)
    rm_hist = RiskMeasure(
        v_pf,
        alpha=alpha_cfg,
        gamma=gamma_cfg)
    VaR_hist = rm_hist.VaR()
    CVaR_hist = rm_hist.CVaR()
    PSRM_hist = rm_hist.power()

    # Parametric (Variance-Covariance)
    rm_param = RiskMeasure(
        var_covar_results,
        alpha=alpha_cfg,
        gamma=gamma_cfg)
    VaR_param = rm_param.VaR()
    CVaR_param = rm_param.CVaR()
    PSRM_param = rm_param.power()

    st.write("### Risk Measure Comparison")
    st.dataframe({
        "Method": ["Historical", "Variance-Covariance"],
        "VaR": [VaR_hist, VaR_param],
        "CVaR": [CVaR_hist, CVaR_param],
        "PowerSpectral": [PSRM_hist, PSRM_param]})

    # Step 4: Visualizations
    st.write("### Scatter: Asset Returns Correlation")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(v1, v2, alpha=0.6)
    ax1.set_xlabel(f"Return {tickers[0]}")
    ax1.set_ylabel(f"Return {tickers[1]}")
    ax1.set_title("Joint Distribution of Asset Returns")
    ax1.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig1)

    st.write("### Portfolio Return Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    hist, bin_edges = np.histogram(v_pf, bins=50, density=True)
    dx = bin_edges[1] - bin_edges[0]
    F_hist = np.cumsum(hist) * dx
    ax2.plot(bin_edges[1:], F_hist, label="Historical Simulation")

    x_range = np.linspace(v_pf.min(), v_pf.max(), 50)
    ax2.plot(
        x_range,
        norm.cdf(x_range, mu_pf, std_pf),
        label="Variance-Covariance Method")

    ax2.set_xlabel("Return")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Historical vs Parametric CDF")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig2)
