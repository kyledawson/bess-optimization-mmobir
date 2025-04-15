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