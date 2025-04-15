# ERCOT BESS Optimization Dashboard

This project provides tools to optimize Battery Energy Storage System (BESS) dispatch for energy arbitrage in the ERCOT Day-Ahead Market (DAM), considering potential Real-Time Market (RTM) price uncertainty.

It includes:
- A Python script (`ercot_api_fetcher.py`) to fetch data from the ERCOT Public API.
- An optimization script (`bess_optimizer.py`) implementing both deterministic and two-stage stochastic optimization models using PuLP.
- A Streamlit dashboard (`streamlit_app.py`) for visualizing optimization inputs and results.
- Supporting scripts for extracting API documentation (`docs/extract_ercot_docs.py`).

## Features

- Fetches ERCOT Day-Ahead Market Settlement Point Prices.
- Optimizes BESS charge/discharge schedule based on DAM prices:
    - **Deterministic Mode:** Assumes perfect foresight of DAM prices.
    - **Stochastic Mode:** Considers RTM price uncertainty using a two-stage stochastic model with generated price scenarios.
- Configurable BESS parameters via a `.env` file.
- Interactive Streamlit dashboard to:
    - Select optimization type (Deterministic/Stochastic).
    - Set target date and settlement point.
    - Adjust stochastic parameters (number of scenarios, noise level).
    - View BESS configuration.
    - Display optimization results (expected/total revenue, charts, schedule table).
    - Visualize generated RTM price scenarios (in stochastic mode).

## Optimization Models

The `bess_optimizer.py` script implements two core optimization strategies:

### 1. Deterministic Optimization (`optimize_day_ahead_arbitrage`)

This model assumes perfect foresight of the Day-Ahead Market (DAM) prices provided as input.

-   **Objective:** Maximize the total profit from energy arbitrage over the 24-hour period. Profit in each hour is calculated as `(Revenue from Discharging) - (Cost of Charging)`, where revenue/cost is determined by the hourly DAM LMP.
    ```
    Maximize Σ [ (DischargePower[h] * DAM_LMP[h]) - (ChargePower[h] * DAM_LMP[h]) ]
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

### 2. Two-Stage Stochastic Optimization (`optimize_two_stage_stochastic`)

This model addresses the uncertainty of Real-Time Market (RTM) prices by optimizing the Day-Ahead schedule while considering multiple possible RTM price scenarios.

-   **Concept:** Uses a two-stage approach:
    -   **Stage 1 (Here-and-Now):** Decides the optimal DAM charge/discharge schedule for the next day *before* knowing the actual RTM prices. This decision is fixed across all scenarios.
    -   **Stage 2 (Wait-and-See):** Models the optimal *adjustments* the BESS would make in the RTM (additional charging/discharging) for *each specific RTM price scenario* that could unfold. These adjustments aim to maximize profit within that scenario, given the fixed DAM schedule.
-   **Objective:** Maximize the *expected* total profit across all scenarios. This includes the profit from the fixed DAM schedule plus the probability-weighted average of the profits from the RTM adjustments made in each scenario.
    ```
    Maximize [ DAM_Profit + Σ (ScenarioProbability[s] * RTM_Profit[s]) ]
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
-   **Output:** The primary result is the optimal Stage 1 DAM schedule, which is designed to be robust across the range of potential RTM outcomes represented by the scenarios.

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
    *   Adjust the default BESS parameters in `.env` if desired.

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
├── bess_optimizer.py   # Core optimization logic (deterministic & stochastic)
├── docs
│   ├── ercot_endpoints.json # Generated API documentation
│   ├── extract_ercot_docs.py # Script to generate endpoints JSON
│   └── pubapi-apim-api.json # ERCOT OpenAPI specification file
├── ercot_api_fetcher.py # Functions to interact with ERCOT API
├── requirements.txt    # Python package dependencies
└── streamlit_app.py    # Streamlit dashboard application
```

## Notes

- The stochastic optimization currently uses a placeholder scenario generation method (adding random noise to DAM prices). For real-world use, replace `generate_rtm_price_scenarios` in `bess_optimizer.py` with a proper forecasting and scenario generation technique.
- The models focus on energy arbitrage and do not currently include Ancillary Service participation or detailed battery degradation modeling. 