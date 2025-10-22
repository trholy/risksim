# 📈 RiskSim – Monte Carlo Risk Simulation using Gaussian Copulas

**RiskSim** is a Python-based simulation and analytics framework for **Monte Carlo–based risk analysis** using **Gaussian copulas**.
It allows users to model **portfolio dependencies**, estimate **risk measures** (VaR, CVaR, power spectral risk measure), and explore **simulation instability** interactively via a **Streamlit web application**.

---

## 🚀 Features

* **Gaussian Copula Simulation** – Model dependent random variables with correlated structures using the Gaussian copula framework.

* **Monte Carlo Engine** – Generate pseudo-random realizations of portfolio components and analyze their joint distributions.

* **Flexible Parameterization** – Define means, standard deviations, and correlations interactively in the Streamlit UI.

* **Interactive Dashboard** – Visualize simulated distributions, copula realizations, and portfolio outcomes directly in the browser.

* **Risk Measure Estimation** – Compute Value-at-Risk (VaR), Conditional VaR (CVaR), and Power Spectral Risk Measure for simulated portfolios.

* **Experiment Mode** – Run multiple simulation batches to analyze **instability** and **variance** of Monte Carlo results under repeated sampling.

---

## 🧮 High-Level Process

### 1. **Variable Specification**

Users define:

* Number of simulation runs (`n_runs`)
* Ranges of random variables (`x_range`, `y_range`)
* Mean vector (`μ`)
* Standard deviations (`σ`)
* Correlation coefficients (`ρ`)

### 2. **Covariance & Cholesky Decomposition**

From the given parameters:

* The **variance–covariance matrix** is constructed.
* A **Cholesky decomposition** is applied to generate correlated random variables.

### 3. **Monte Carlo Simulation**

The Gaussian copula–based simulation proceeds as follows:

1. Generate independent uniform pseudo-random numbers.
2. Transform them to independent standard normal variates.
3. Introduce correlation using Cholesky decomposition.
4. Transform correlated normals to correlated uniforms (the copula).
5. Apply user-defined marginals to obtain final dependent realizations ( X ) and ( Y ).
6. Compute the **portfolio sum** ( S = X + Y ).

### 4. **Visualization**

RiskSim includes multiple visualization functions:

* **Scatter Plots:**

  * Dependent bivariate normal realizations
  * Gaussian copula realizations
  * Uniformly distributed realizations with copula dependency

* **Histograms & CDFs:**

  * Portfolio sum distributions
  * Cumulative distributions illustrating dependency effects

### 5. **Risk Measure Estimation**

RiskSim integrates with a `RiskMeasure` class that calculates:

* **VaR (Value at Risk)**
* **CVaR (Conditional Value at Risk)**
* **PSRM (Power Spectral Risk Measure)**

These are evaluated for each simulation run, allowing users to assess **stability** and **sensitivity** across multiple Monte Carlo realizations.

### 6. **Instability Analysis**

Monte Carlo simulations can exhibit significant result variability when the number of samples is low.
RiskSim allows you to configure:

* Number of simulation runs
* Number of samples per run

to study convergence and instability effects interactively.

---

## ▶️ Run the Streamlit Application

1. **Clone the repository**

   ```bash
   git clone https://github.com/trholy/risksim.git
   cd risksim
   ```

2. **Build and run docker container**

   ```bash
   docker-compose up --build
   ```

3. **Open in your browser:**
   The app will launch at [http://localhost:8501](http://localhost:8501)
---

## ⚙️ Example Configuration

**Simulation parameters (UI configurable):**

```python
n_runs = 10000
x_range = (10.0, 20.0)
y_range = (8.0, 22.0)
mu = [2.0, 3.0]
std = [2.0, 3.0]
corr = [0.0]
```

**Experiment parameters (UI configurable):**

```python
runs = 10
sim_samples = 100
alpha = 0.1
gamma = 0.5
```

---

## 🧠 Core Concepts

### **Gaussian Copula**

The **Gaussian Copula** links multiple random variables through a correlation matrix while preserving their marginal distributions.
It enables realistic modeling of portfolio dependencies even when marginals are non-Gaussian.

### **Cholesky Decomposition**

Used to induce correlation between independent standard normal variables:
$$Z_{\text{corr}} = Z_{\text{indep}} \cdot L^T$$
where ( L ) is the lower-triangular matrix from Cholesky decomposition.

### **Risk Measures**

* **Value-at-Risk (VaR):**
  The loss threshold exceeded with probability α.

* **Conditional Value-at-Risk (CVaR):**
  The expected loss given that VaR has been exceeded.

* **Power Spectral Risk Measure (PSRM):**
  A nonlinear risk measure incorporating a power factor γ.

---

## 📊 Example Visualizations

* **Dependent Bivariate Normal Realizations**
* **Gaussian Copula Realizations (Uniform space)**
* **Uniform Realizations with Copula Dependency**
* **Portfolio Sum Histogram**
* **Cumulative Distribution Function (CDF)**
* **Monte Carlo Instability Across Multiple Runs**

Each visualization helps illustrate both **dependency structures** and **portfolio outcome distributions**.

---

## 📂 Project Structure

```
risksim
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── datasets
├── docker-compose.yml
├── pyproject.toml
├── setup.py
├── src
│   └── risksim
│       ├── __init__.py
│       ├── copula
│       │   └── copula.py
│       ├── experiment
│       │   └── experiment.py
│       ├── plot
│       │   └── plotting.py
│       ├── risk
│       │   └── risk.py
│       └── utils
│           └── utils.py
└── streamlit-app
    ├── app.py
    └── requirements.txt
```

---

## 📜 License

This project is released under the **[MIT License](LICENSE)**.
You are free to use, modify, and distribute it with proper attribution.
