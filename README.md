# Jump-Diffusion Hedging Simulator & Greek Attribution Engine (`jump-diffusion-hedger`)

A production-grade, vectorized non-linear derivatives risk engine and dynamic hedging simulation framework. Evaluates the breakdown of discrete Black-Scholes delta-gamma hedging under Merton jump-diffusion processes, stochastic volatility regimes, and discrete rebalancing frictions.

---

## 1. Architectural & Engineering Directives

- **Runtime Target:** Python 3.12+ (deterministic numerical pipelines).
- **Strict Typing:** PEP 585/604 lowercase syntax, PEP 646 / `numpy.typing.NDArray[np.float64]` for state spaces, price paths, and Greek tensors.
- **Anti-Loop Mandate:** Zero procedural loops over asset paths or simulation time steps. All path trajectories generated via vector-matrix broadcasting and prefix scans (`np.cumsum`).
- **Memory Profiling:** Pre-allocated C-contiguous memory blocks; vectorized 2D/3D tensor operations for multi-asset or multi-path Greek attribution.
- **Decoupled Architecture:** Stateless pricing kernel accepting strictly typed array inputs, completely isolated from visualization or data frame metadata.

---

## 2. Quantitative Mechanics & Scope

### A. Asset Price Trajectory Engines
- **Merton Jump-Diffusion Process:**
  $$dS_t = (r - q - \lambda k) S_t dt + \sigma S_t dW_t + (J - 1) S_t dN_t$$
  where $N_t$ is a homogeneous Poisson process with intensity $\lambda$, and $\ln(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$.
- **Vectorized Discretization:** Simultaneous generation of $M$ paths over $N$ time increments using compound Poisson random variates and standard normal innovations.

### B. Analytical Greek Tensors & Hedging Friction
- **SIMD Greek Evaluation:** Analytic Black-Scholes pricing and Greeks (Delta $\Delta$, Gamma $\Gamma$, Vega $\mathcal{V}$, Theta $\Theta$, Rho $\rho$) evaluated simultaneously across 2D/3D meshgrids $(S, t)$ using `scipy.special.ndtr` without Python-level iteration.
- **Dynamic Hedging Breakdown:**
  - Tracking error simulation across discrete rebalancing intervals (daily, intra-day, weekly).
  - Transaction cost impact via proportional and fixed bid-ask spread models.
- **PnL Attribution Formulation:**
  Decomposes realized portfolio PnL at each interval $t \to t + \Delta t$ into:
  $$\Delta \Pi \approx \left( \Theta + \frac{1}{2} \Gamma S^2 \sigma_{\text{realized}}^2 \right) \Delta t + \mathcal{V} \Delta \sigma + \text{Jump Residual}$$
  contrasting theoretical replication against discrete jump gaps.

---

## 3. Directory Layout

```text
jump-diffusion-hedger/
├── pyproject.toml
├── README.md
├── src/
│   └── hedging_engine/
│       ├── __init__.py
│       ├── types.py          # Custom array annotations, Greek dataclasses
│       ├── processes.py      # Vectorized GBM and Merton jump-diffusion generators
│       ├── analytics.py      # Closed-form European BSM pricer and Greek tensors
│       ├── hedging.py        # Discrete rebalancing simulator and cost models
│       └── attribution.py    # PnL Taylor decomposition (Gamma bleed vs Jump error)
└── tests/
    ├── test_processes.py
    └── test_attribution.py
```
