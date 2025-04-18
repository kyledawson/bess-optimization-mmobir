# ERCOT BESS Optimization Dashboard

This project provides tools to optimize Battery Energy Storage System (BESS) dispatch for energy arbitrage in the ERCOT Day-Ahead Market (DAM), considering potential Real-Time Market (RTM) price uncertainty.

It includes:
- A Python script (`ercot_api_fetcher.py`) to fetch data from the ERCOT Public API.
- An optimization script (`bess_optimizer.py`) implementing both deterministic and two-stage stochastic optimization models using PuLP.
- **Degradation Cost Modeling:** Incorporates a simplified linear cost of degradation based on battery cost, cycle life, and energy throughput.
- A Streamlit dashboard (`streamlit_app.py`) for visualizing optimization inputs and results.
- Supporting scripts for extracting API documentation (`docs/extract_ercot_docs.py`).

## Features

- Fetches ERCOT Day-Ahead Market Settlement Point Prices.
- Optimizes BESS charge/discharge schedule based on DAM prices, maximizing **Net Revenue** (Gross Revenue - Degradation Cost):
    - **Deterministic Mode:** Assumes perfect foresight of DAM prices.
    - **Stochastic Mode:** Considers RTM price uncertainty using a two-stage stochastic model with generated price scenarios.
- Configurable BESS parameters via a `.env` file (or `values.md` as reference).
- **Simplified Degradation Cost Model:** Calculates an average cost per MWh cycled based on total battery cost and cycle life at 100% DoD.
- Interactive Streamlit dashboard to:
    - Select optimization type (Deterministic/Stochastic).
    - Set target date and settlement point.
    - Adjust stochastic parameters (number of scenarios, noise level).
    - View BESS configuration.
    - Display optimization results (Net Revenue, Gross Revenue, Degradation Cost, charts, schedule table).
    - Visualize generated RTM price scenarios (in stochastic mode).
    - Explore detailed explanations of the models and parameters in the "BESS Optimization Guide" tab.

## Optimization Models

The `bess_optimizer.py` script implements two core optimization strategies:

### 1. Deterministic Optimization (`optimize_day_ahead_arbitrage`)

This model assumes perfect foresight of the Day-Ahead Market (DAM) prices provided as input.

-   **Objective:** Maximize the total **Net Revenue** over the 24-hour period. Net Revenue = `(Gross Revenue from Arbitrage) - (Calculated Degradation Cost)`.
    ```
    Maximize Σ [ (DischargePower[h] * DAM_LMP[h]) - (ChargePower[h] * DAM_LMP[h]) ] - Σ DegradationCost[h]
    ```
-   **Key Decision Variables:**
    -   `ChargePower[h]`: Power (MW) used to charge the battery in hour `h`.
    -   `DischargePower[h]`: Power (MW) delivered by discharging the battery in hour `h`.
    -   `SoC[h]`: State of Charge (MWh) of the battery at the *end* of hour `h`.
-   **Key Constraints:**
    -   **Power Limits:** Charge/Discharge power cannot exceed the BESS maximum ratings (`max_charge_power_mw`, `max_discharge_power_mw`).
    -   **SoC Limits:** The battery's state of charge must remain within its minimum and maximum MWh limits (`min_soc_mwh`, `max_soc_mwh`).
    -   **Energy Balance:** The SoC at the end of an hour depends on the previous hour's SoC, the energy added during charging (adjusted for charge efficiency), and the energy removed during discharging (adjusted for discharge efficiency).
        ```
        SoC[h+1] = SoC[h] + (ChargePower[h] * ChargeEfficiency) - (DischargePower[h] / DischargeEfficiency)
        ```
    -   **No Simultaneous Charge/Discharge:** Ensures the battery is either charging, discharging, or idle in any given hour.
    -   **Degradation Cost:** Calculated based on energy throughput and the average cost per MWh cycled (see Degradation Model section).

### 2. Two-Stage Stochastic Optimization (`optimize_two_stage_stochastic`)

This model addresses the uncertainty of Real-Time Market (RTM) prices by optimizing the Day-Ahead schedule while considering multiple possible RTM price scenarios.

-   **Concept:** Uses a two-stage approach:
    -   **Stage 1 (Here-and-Now):** Decides the optimal DAM charge/discharge schedule for the next day *before* knowing the actual RTM prices. This decision is fixed across all scenarios.
    -   **Stage 2 (Wait-and-See):** Models the optimal *adjustments* the BESS would make in the RTM (additional charging/discharging) for *each specific RTM price scenario* that could unfold. These adjustments aim to maximize profit within that scenario, given the fixed DAM schedule.
-   **Objective:** Maximize the *expected* **Net Revenue** across all scenarios. This includes the profit from the fixed DAM schedule plus the probability-weighted average of the profits from the RTM adjustments, minus the expected degradation cost.
    ```
    Maximize [ DAM_Profit + Σ (ScenarioProbability[s] * RTM_Profit[s]) ] - Expected_DegradationCost
    ```
-   **Scenarios:** The model requires a set of potential RTM price scenarios (`rtm_scenarios`), each with an associated probability. *(Currently, a placeholder function `generate_rtm_price_scenarios` creates these by adding random noise to the DAM prices).* 
-   **Key Decision Variables:**
    -   **Stage 1:** `DAM_ChargePower[h]`, `DAM_DischargePower[h]` (one value per hour).
    -   **Stage 2:** `RTM_ChargePower[h, s]`, `RTM_DischargePower[h, s]`, `SoC[h, s]` (values exist for each hour `h` and each scenario `s`).
-   **Key Constraints:**
    -   **Stage 1 Power Limits:** DAM charge/discharge is limited.
    -   **Stage 2 Constraints (per scenario):**
        -   **Total Power Limits:** The *sum* of DAM power and RTM adjustment power (charge or discharge) in any hour `h` under scenario `s` cannot exceed the BESS ratings.
        -   **SoC Limits:** SoC must remain within bounds in every hour for every scenario.
        -   **Energy Balance:** The SoC transition within each scenario `s` considers both the committed DAM actions and the specific RTM adjustments for that scenario.
            ```
            SoC[h+1, s] = SoC[h, s] + (TotalCharge[h, s] * ChargeEff) - (TotalDischarge[h, s] / DischargeEff)
            ```
    -   **Degradation Cost (per scenario):** Calculated based on energy throughput in each scenario and the average cost per MWh cycled.
-   **Output:** The primary result is the optimal Stage 1 DAM schedule, which is designed to be robust across the range of potential RTM outcomes represented by the scenarios, considering both energy arbitrage revenue and degradation costs.

### 3. Degradation Cost Model (`load_bess_config`)

To account for the cost associated with battery wear-and-tear, a simplified linear degradation cost model is implemented:

-   **Calculation:** An average cost per MWh cycled (`avg_degradation_cost_per_mwh`) is calculated based on the total battery cost and its lifetime energy throughput (derived from capacity and cycle life at 100% DoD).
    ```python
    # Simplified calculation in load_bess_config
    total_lifetime_throughput_mwh = config["cycle_life_at_100pct_dod"] * config["capacity_mwh"]
    config["avg_degradation_cost_per_mwh"] = config["total_battery_cost"] / total_lifetime_throughput_mwh
    ```
-   **Application:** This average cost is multiplied by the energy throughput (approximated by charge power) in each hour to estimate the hourly degradation cost, which is then subtracted from the gross revenue in the objective function.
    ```python
    # Simplified constraint in optimization models
    energy_throughput[h] == charge_power[h]
    degradation_cost[h] >= energy_throughput[h] * avg_cost_per_mwh_cycled
    ```
-   **Linearity:** This approach keeps the optimization problem linear, making it solvable with standard LP solvers. More complex, non-linear degradation models (e.g., considering DoD, temperature, C-rate explicitly in the cost function) would require more advanced solvers.
-   **Parameters:** The key `.env` variables influencing this cost are:
    -   `BESS_COST_USD_PER_KWH`
    -   `BESS_CAPACITY_MWH`
    -   `BESS_CYCLE_LIFE_100PCT_DOD`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MahmoudMobir/bess-optimization-ercot.git
    cd bess-optimization-ercot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment (example for bash/zsh)
    source venv/bin/activate
    # Or for Windows Command Prompt:
    # venv\Scripts\activate.bat
    # Or for Windows PowerShell:
    # .\venv\Scripts\Activate.ps1 
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your actual ERCOT API credentials:
        ```
        ERCOT_API_KEY=YOUR_ERCOT_API_KEY_HERE
        ERCOT_USERNAME=your_ercot_api_username
        ERCOT_PASSWORD=your_ercot_api_password
        ```
    *   Adjust the default BESS parameters in `.env` if desired. Refer to `.env.example` or `values.md` for required parameters, including the degradation inputs.

## Usage

### Streamlit Dashboard (Recommended)

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

This will open the interactive dashboard in your browser. Use the sidebar to configure parameters and run the optimization.

### Command-Line Optimizer (for testing)

You can also run the optimizer directly from the command line. This is useful for testing the core logic. By default, it runs tests for both deterministic and stochastic modes using parameters defined within the script's `if __name__ == "__main__":` block.

```bash
python bess_optimizer.py
```

## Project Structure

```
.
├── .env                # Local environment variables (ignored by git)
├── .env.example        # Example environment variables template
├── .gitignore          # Files ignored by git
├── README.md           # This file
├── TODO.md             # Future development roadmap
├── bess_optimizer.py   # Core optimization logic (deterministic & stochastic)
├── docs
│   ├── ercot_endpoints.json # Generated API documentation
│   ├── extract_ercot_docs.py # Script to generate endpoints JSON
│   └── pubapi-apim-api.json # ERCOT OpenAPI specification file
├── ercot_api_fetcher.py # Functions to interact with ERCOT API
├── requirements.txt    # Python package dependencies
└── streamlit_app.py    # Streamlit dashboard application
└── values.md           # Reference values for BESS parameters
```

## Notes

- The stochastic optimization currently uses a placeholder scenario generation method (adding random noise to DAM prices). For real-world use, replace `generate_rtm_price_scenarios` in `bess_optimizer.py` with a proper forecasting and scenario generation technique.
- The degradation model is a simplified linear approximation. See `TODO.md` for potential enhancements.
