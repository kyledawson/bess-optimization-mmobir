import os
import pandas as pd
import pulp
from datetime import datetime, timedelta, timezone
from math import sqrt
from dotenv import load_dotenv
import traceback # For detailed error logging
import random # For scenario generation placeholder

# Import the ERCOT API fetcher
from ercot_api_fetcher import fetch_day_ahead_prices, ERCOTAPIError

# Load environment variables (including BESS parameters)
load_dotenv()

# --- Configuration Loading ---
def load_bess_config():
    """Loads BESS configuration from environment variables."""
    try:
        config = {
            "capacity_mwh": float(os.getenv("BESS_CAPACITY_MWH", 10)),
            "max_charge_power_mw": float(os.getenv("BESS_MAX_CHARGE_POWER_MW", 5)),
            "max_discharge_power_mw": float(os.getenv("BESS_MAX_DISCHARGE_POWER_MW", 5)),
            "min_soc_percent": float(os.getenv("BESS_MIN_SOC_PERCENT", 10)),
            "max_soc_percent": float(os.getenv("BESS_MAX_SOC_PERCENT", 90)),
            "round_trip_efficiency_percent": float(os.getenv("BESS_ROUND_TRIP_EFFICIENCY_PERCENT", 85)),
            "initial_soc_percent": float(os.getenv("BESS_INITIAL_SOC_PERCENT", 50))
        }
        # Convert percentages to decimals
        config["min_soc_mwh"] = config["capacity_mwh"] * config["min_soc_percent"] / 100.0
        config["max_soc_mwh"] = config["capacity_mwh"] * config["max_soc_percent"] / 100.0
        config["initial_soc_mwh"] = config["capacity_mwh"] * config["initial_soc_percent"] / 100.0
        config["charge_efficiency"] = sqrt(config["round_trip_efficiency_percent"] / 100.0)
        config["discharge_efficiency"] = sqrt(config["round_trip_efficiency_percent"] / 100.0)
        return config
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error parsing BESS configuration from .env file: {e}. Ensure all BESS parameters are valid numbers.")

# --- Deterministic Optimization Logic ---
def optimize_day_ahead_arbitrage(bess_config: dict, prices_df: pd.DataFrame) -> tuple:
    """
    Optimizes BESS dispatch for day-ahead energy arbitrage using PuLP (Deterministic).
    Assumes perfect foresight based on the input prices_df.
    """
    # --- Model Setup ---
    prob = pulp.LpProblem("BESS_DayAhead_Arbitrage_Deterministic", pulp.LpMaximize)
    hours = range(len(prices_df))

    # --- Decision Variables ---
    charge_power = pulp.LpVariable.dicts("ChargePower", hours, lowBound=0, upBound=bess_config["max_charge_power_mw"], cat='Continuous')
    discharge_power = pulp.LpVariable.dicts("DischargePower", hours, lowBound=0, upBound=bess_config["max_discharge_power_mw"], cat='Continuous')
    soc_mwh = pulp.LpVariable.dicts("SoC", range(len(hours) + 1), lowBound=bess_config["min_soc_mwh"], upBound=bess_config["max_soc_mwh"], cat='Continuous')
    is_charging = pulp.LpVariable.dicts("IsCharging", hours, cat='Binary')

    # --- Objective Function ---
    prob += pulp.lpSum(
        (discharge_power[h] * prices_df.loc[h, 'LMP'] - charge_power[h] * prices_df.loc[h, 'LMP']) 
        for h in hours
    ), "Total_Revenue"

    # --- Constraints ---
    prob += soc_mwh[0] == bess_config["initial_soc_mwh"], "Initial_SoC"
    discharge_efficiency_inverse = 1.0 / bess_config["discharge_efficiency"]
    for h in hours:
        prob += soc_mwh[h+1] == soc_mwh[h] + (charge_power[h] * bess_config["charge_efficiency"]) - (discharge_power[h] * discharge_efficiency_inverse), f"SoC_Balance_Hour_{h}"
    M = max(bess_config["max_charge_power_mw"], bess_config["max_discharge_power_mw"]) * 1.1
    for h in hours:
        prob += charge_power[h] <= M * is_charging[h], f"Charge_Indicator_{h}"
        prob += discharge_power[h] <= M * (1 - is_charging[h]), f"Discharge_Indicator_{h}"

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=False) 
    prob.solve(solver)

    # --- Process Results ---
    status = pulp.LpStatus[prob.status]
    if status == 'Optimal':
        schedule = []
        total_revenue = pulp.value(prob.objective)
        for h in hours:
            schedule.append({
                'HourEnding': prices_df.loc[h, 'HourEnding'],
                'LMP': prices_df.loc[h, 'LMP'],
                'ChargePower_MW': charge_power[h].varValue,
                'DischargePower_MW': discharge_power[h].varValue,
                'SoC_MWh_End': soc_mwh[h+1].varValue,
                'SoC_Percent_End': (soc_mwh[h+1].varValue / bess_config["capacity_mwh"]) * 100
            })
        schedule_df = pd.DataFrame(schedule)
        return status, schedule_df, total_revenue
    else:
        print(f"Deterministic optimization failed with status: {status}")
        return status, None, None

# --- NEW: Scenario Generation Placeholder ---
def generate_rtm_price_scenarios(dam_prices_df: pd.DataFrame, num_scenarios: int, noise_std_dev: float = 5.0) -> dict:
    """
    Placeholder function to generate RTM price scenarios by adding noise to DAM prices.
    
    Args:
        dam_prices_df: DataFrame with DAM 'HourEnding' and 'LMP'.
        num_scenarios: Number of scenarios to generate.
        noise_std_dev: Standard deviation of the normal noise added ($/MWh).

    Returns:
        A dictionary where keys are scenario indices (0 to num_scenarios-1) 
        and values are pandas Series of RTM prices for that scenario, indexed like dam_prices_df.
    """
    scenarios = {}
    if 'LMP' not in dam_prices_df.columns:
         raise ValueError("DAM prices DataFrame must contain 'LMP' column.")
         
    for s in range(num_scenarios):
        # Add random noise (mean 0) to DAM prices
        # Ensure prices don't go below a certain floor (e.g., $0 or a typical floor)
        noise = [random.gauss(0, noise_std_dev) for _ in range(len(dam_prices_df))]
        # Use .copy() to avoid modifying the original DAM LMP Series
        rtm_lmp = dam_prices_df['LMP'].copy() + noise 
        rtm_lmp = rtm_lmp.clip(lower=0) # Example: prevent negative prices
        scenarios[s] = rtm_lmp.reset_index(drop=True) # Ensure clean index for .loc access later
        
    print(f"Generated {num_scenarios} RTM price scenarios with std dev {noise_std_dev}.")
    return scenarios

# --- NEW: Two-Stage Stochastic Optimization Logic ---
def optimize_two_stage_stochastic(bess_config: dict, dam_prices_df: pd.DataFrame, rtm_scenarios: dict) -> tuple:
    """
    Optimizes BESS dispatch using a two-stage stochastic model.
    Stage 1: DAM Charge/Discharge decisions (here-and-now).
    Stage 2: RTM Charge/Discharge adjustments & SoC (wait-and-see, per scenario).

    Args:
        bess_config: Dictionary containing BESS parameters.
        dam_prices_df: DataFrame with DAM 'HourEnding' and 'LMP'.
        rtm_scenarios: Dictionary {scenario_index: pd.Series(RTM_LMP)}

    Returns:
        A tuple containing: (status, dam_schedule_df, expected_revenue)
        status (str): PuLP solver status.
        dam_schedule_df (pd.DataFrame or None): DataFrame with optimal Stage 1 (DAM) schedule.
        expected_revenue (float or None): Calculated maximum expected revenue across scenarios.
    """
    num_scenarios = len(rtm_scenarios)
    if num_scenarios == 0:
        raise ValueError("rtm_scenarios dictionary cannot be empty.")
        
    # Assume equal probability for simplicity
    scenario_probability = 1.0 / num_scenarios 
    scenarios_idx = list(rtm_scenarios.keys())

    # --- Model Setup ---
    prob = pulp.LpProblem("BESS_TwoStage_Stochastic", pulp.LpMaximize)
    hours = range(len(dam_prices_df))

    # --- Decision Variables ---
    # Stage 1 (DAM - single decision across all scenarios)
    dam_charge_power = pulp.LpVariable.dicts("DAM_ChargePower", hours, lowBound=0, upBound=bess_config["max_charge_power_mw"], cat='Continuous')
    dam_discharge_power = pulp.LpVariable.dicts("DAM_DischargePower", hours, lowBound=0, upBound=bess_config["max_discharge_power_mw"], cat='Continuous')
    # Binary DAM variable to prevent simultaneous charge/discharge
    is_dam_charging = pulp.LpVariable.dicts("Is_DAM_Charging", hours, cat='Binary')

    # Stage 2 (RTM Adjustments & SoC - per scenario)
    # Using tuples (hour, scenario) as keys
    rtm_charge_power = pulp.LpVariable.dicts("RTM_ChargePower", [(h, s) for h in hours for s in scenarios_idx], lowBound=0, cat='Continuous')
    rtm_discharge_power = pulp.LpVariable.dicts("RTM_DischargePower", [(h, s) for h in hours for s in scenarios_idx], lowBound=0, cat='Continuous')
    # SoC exists for hour 0 through final hour, per scenario
    soc_mwh = pulp.LpVariable.dicts("SoC", [(t, s) for t in range(len(hours) + 1) for s in scenarios_idx], 
                                   lowBound=bess_config["min_soc_mwh"], upBound=bess_config["max_soc_mwh"], cat='Continuous')

    # --- Objective Function (Maximize Expected Profit) ---
    # Expected Profit = DAM Profit + Sum over scenarios [ Prob(s) * RTM Profit(s) ]
    dam_profit = pulp.lpSum(
        (dam_discharge_power[h] - dam_charge_power[h]) * dam_prices_df.loc[h, 'LMP'] 
        for h in hours
    )
    
    expected_rtm_profit = pulp.lpSum(
        scenario_probability * (rtm_discharge_power[h, s] - rtm_charge_power[h, s]) * rtm_scenarios[s].iloc[h] # Use .iloc for Series access by position
        for h in hours for s in scenarios_idx
    )

    prob += dam_profit + expected_rtm_profit, "Total_Expected_Revenue"

    # --- Constraints ---
    # ** Stage 1 Constraints **
    # Prevent Simultaneous DAM Charge and Discharge
    M_dam = max(bess_config["max_charge_power_mw"], bess_config["max_discharge_power_mw"]) * 1.1 # Big-M
    for h in hours:
        prob += dam_charge_power[h] <= M_dam * is_dam_charging[h], f"DAM_Charge_Indicator_{h}"
        prob += dam_discharge_power[h] <= M_dam * (1 - is_dam_charging[h]), f"DAM_Discharge_Indicator_{h}"

    # ** Stage 2 Constraints (Applied PER SCENARIO) **
    discharge_efficiency_inverse = 1.0 / bess_config["discharge_efficiency"]
    charge_efficiency = bess_config["charge_efficiency"]
    
    for s in scenarios_idx:
        # Initial SoC (same for all scenarios)
        prob += soc_mwh[(0, s)] == bess_config["initial_soc_mwh"], f"Initial_SoC_Scenario_{s}"

        for h in hours:
            # Total power in the hour (DAM + RTM adjustment)
            total_charge = dam_charge_power[h] + rtm_charge_power[h, s]
            total_discharge = dam_discharge_power[h] + rtm_discharge_power[h, s]

            # SoC Balance for this scenario
            # Ensure correct tuple indexing for SoC
            prob += soc_mwh[(h+1, s)] == soc_mwh[(h, s)] + (total_charge * charge_efficiency) - (total_discharge * discharge_efficiency_inverse), f"SoC_Balance_Hour_{h}_Scenario_{s}"

            # Total Power Limits for this scenario
            prob += total_charge <= bess_config["max_charge_power_mw"], f"MaxChargeLimit_Hour_{h}_Scenario_{s}"
            prob += total_discharge <= bess_config["max_discharge_power_mw"], f"MaxDischargeLimit_Hour_{h}_Scenario_{s}"
            
            # Prevent simultaneous RTM charge and discharge *adjustments*? 
            # Adding constraints to prevent simultaneous RTM adjustments might be complex
            # and depends on market rules. Let's assume net RTM adjustment is possible for now.
            # If needed, add binary vars for RTM charge/discharge per scenario & link them.

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=False) 
    prob.solve(solver)

    # --- Process Results ---
    status = pulp.LpStatus[prob.status]
    if status == 'Optimal':
        # Extract the Stage 1 (DAM) schedule - this is the primary output
        dam_schedule = []
        expected_revenue = pulp.value(prob.objective)
        for h in hours:
            dam_schedule.append({
                'HourEnding': dam_prices_df.loc[h, 'HourEnding'],
                'DAM_LMP': dam_prices_df.loc[h, 'LMP'], # Include DAM price for context
                'DAM_ChargePower_MW': dam_charge_power[h].varValue,
                'DAM_DischargePower_MW': dam_discharge_power[h].varValue,
                # Add RTM price stats if useful? e.g. avg RTM price for this hour
                'Avg_RTM_LMP_Scenario': sum(rtm_scenarios[s].iloc[h] for s in scenarios_idx) / num_scenarios
            })
        dam_schedule_df = pd.DataFrame(dam_schedule)
        
        # Note: We are not returning the detailed Stage 2 variables 
        return status, dam_schedule_df, expected_revenue
    else:
        print(f"Stochastic optimization failed with status: {status}")
        return status, None, None

# --- Data Fetching and Processing Function ---
def fetch_and_process_dam_prices(settlement_point: str, target_date: datetime.date) -> pd.DataFrame:
    """
    Fetches and processes Day-Ahead Market prices for the given date and settlement point.
    Returns a cleaned DataFrame with 'HourEnding' (datetime) and 'LMP' (float).
    Raises ERCOTAPIError or ValueError on failure.
    """
    print(f"Fetching Day-Ahead prices for {target_date} at {settlement_point}...")
    price_data = fetch_day_ahead_prices(settlement_point=settlement_point, delivery_date=target_date, debug=False)

    if not price_data or not price_data.get("data") or not price_data.get("fields"):
        raise ValueError("Failed to fetch price data, or 'data'/'fields' key missing in response.")

    # Extract column names
    try:
        column_names = [field['name'] for field in price_data['fields']]
    except (TypeError, KeyError) as e:
        raise ValueError(f"Error parsing 'fields' key: {e}")

    # Create DataFrame
    prices = pd.DataFrame(price_data["data"], columns=column_names)

    # Validation and Processing
    required_date_col = 'deliveryDate'
    required_time_col = 'hourEnding'
    required_price_col = 'settlementPointPrice'

    missing_cols = [col for col in [required_date_col, required_time_col, required_price_col] if col not in prices.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols}. Available: {list(prices.columns)}")

    # Combine Date and Hour
    try:
        datetime_str = prices[required_date_col].astype(str) + ' ' + prices[required_time_col].astype(str)
        datetime_str_corrected = datetime_str.str.replace(' 24:00', ' 00:00')
        prices['HourEnding'] = pd.to_datetime(datetime_str_corrected, format='%Y-%m-%d %H:%M', errors='coerce')
        mask_hour_24 = datetime_str.str.contains(' 24:00')
        prices.loc[mask_hour_24, 'HourEnding'] += timedelta(days=1)
    except Exception as e:
        raise ValueError(f"Error creating timestamp: {e}")

    prices.dropna(subset=['HourEnding'], inplace=True)

    # Convert price to numeric
    prices['LMP'] = pd.to_numeric(prices[required_price_col], errors='coerce')
    prices.dropna(subset=['LMP'], inplace=True)

    # Select and sort
    prices = prices[['HourEnding', 'LMP']].sort_values(by='HourEnding').reset_index(drop=True)

    if prices.empty:
        raise ValueError("No valid price data found after processing.")

    print(f"Successfully fetched and processed {len(prices)} hourly prices.")
    return prices

# --- UPDATED: Main Runner Function ---
def run_optimization_pipeline(
    target_date: datetime.date, 
    settlement_point: str,
    optimization_type: str = 'deterministic', # 'deterministic' or 'stochastic'
    num_scenarios: int = 5, # Used only if type is stochastic
    noise_std_dev: float = 5.0 # Used only if type is stochastic
    ) -> dict:
    """
    Runs the full pipeline: load config, fetch/process data, run optimization.
    Can run either deterministic or two-stage stochastic optimization.
    
    Args:
        target_date: The date for which to optimize.
        settlement_point: The ERCOT settlement point.
        optimization_type: 'deterministic' or 'stochastic'.
        num_scenarios: Number of RTM scenarios for stochastic mode.
        noise_std_dev: Standard deviation for RTM scenario price noise.

    Returns:
        A dictionary containing results:
        {
            'success': bool,
            'optimization_type': str,
            'bess_config': dict | None,
            'dam_prices_df': pd.DataFrame | None,
            'rtm_scenario_prices': dict | None, # Dict {scen_idx: pd.Series(LMP)}
            'status': str | None,
            'schedule_df': pd.DataFrame | None, # DAM schedule for stochastic, full schedule for deterministic
            'total_revenue': float | None, # Actual revenue for deterministic
            'expected_revenue': float | None, # Expected revenue for stochastic
            'error_message': str | None
        }
    """
    results = {
        'success': False,
        'optimization_type': optimization_type,
        'bess_config': None,
        'dam_prices_df': None,
        'rtm_scenario_prices': None, 
        'status': None,
        'schedule_df': None, 
        'total_revenue': None, 
        'expected_revenue': None, 
        'error_message': None
    }
    try:
        # Load BESS configuration
        bess_params = load_bess_config()
        results['bess_config'] = bess_params

        # Fetch and Process DAM Data (needed for both types)
        dam_prices_df = fetch_and_process_dam_prices(settlement_point, target_date)
        results['dam_prices_df'] = dam_prices_df
        
        if optimization_type == 'stochastic':
            print(f"\nRunning Two-Stage Stochastic Optimization with {num_scenarios} scenarios...")
            # Generate RTM Scenarios (Placeholder)
            rtm_scenarios = generate_rtm_price_scenarios(dam_prices_df, num_scenarios, noise_std_dev)
            results['rtm_scenario_prices'] = rtm_scenarios # Store for potential display

            # Run Stochastic Optimization
            opt_status, dam_schedule_df, expected_revenue = optimize_two_stage_stochastic(bess_params, dam_prices_df, rtm_scenarios)
            results['status'] = opt_status
            results['schedule_df'] = dam_schedule_df # This is the DAM schedule
            results['expected_revenue'] = expected_revenue
            if opt_status == 'Optimal':
                results['success'] = True
            else:
                 results['error_message'] = f"Stochastic optimization finished with status: {opt_status}"

        elif optimization_type == 'deterministic':
             print("\nRunning Deterministic Optimization...")
             # Run Deterministic Optimization
             opt_status, schedule_df, total_revenue = optimize_day_ahead_arbitrage(bess_params, dam_prices_df)
             results['status'] = opt_status
             results['schedule_df'] = schedule_df # Full schedule
             results['total_revenue'] = total_revenue
             if opt_status == 'Optimal':
                 results['success'] = True
             else:
                  results['error_message'] = f"Deterministic optimization finished with status: {opt_status}"
        else:
             raise ValueError(f"Unknown optimization_type: {optimization_type}")
            
    except (ERCOTAPIError, ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error during optimization pipeline: {e}")
        results['error_message'] = str(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc()) 
        results['error_message'] = f"An unexpected error occurred: {e}"
        
    return results

# --- Main Execution Block (UPDATED for testing both modes) ---
if __name__ == "__main__":
    print("Running BESS Optimizer (Command Line)...")

    # Example usage parameters
    target_date_cli = datetime.now(timezone.utc).date() - timedelta(days=1)
    settlement_point_cli = "HB_NORTH"
    
    # --- Test Deterministic ---
    print("\n--- TESTING DETERMINISTIC ---")
    run_results_det = run_optimization_pipeline(
        target_date_cli, 
        settlement_point_cli, 
        optimization_type='deterministic'
    )
    if run_results_det['success']:
        print(f"Deterministic Status: {run_results_det['status']}")
        print(f"Deterministic Revenue: ${run_results_det['total_revenue']:.2f}")
        # Optional: Print schedule head
        # schedule_display = run_results_det['schedule_df'].copy()
        # for col, fmt in [('LMP', '{:.2f}'), ('ChargePower_MW', '{:.3f}'), ('DischargePower_MW', '{:.3f}'), ('SoC_MWh_End', '{:.3f}'), ('SoC_Percent_End', '{:.1f}')]:
        #      if col in schedule_display.columns: schedule_display[col] = schedule_display[col].map(lambda x: fmt.format(x) if pd.notna(x) else 'NaN')
        # print("Deterministic Schedule (Head):")
        # print(schedule_display.head().to_string())
    else:
        print(f"Deterministic Failed: {run_results_det['error_message']}")

    # --- Test Stochastic ---
    print("\n--- TESTING STOCHASTIC ---")
    run_results_stoch = run_optimization_pipeline(
        target_date_cli, 
        settlement_point_cli, 
        optimization_type='stochastic',
        num_scenarios=5, # Example: 5 scenarios
        noise_std_dev=10.0 # Example: +/- $10/MWh noise std dev
    )
    if run_results_stoch['success']:
        print(f"Stochastic Status: {run_results_stoch['status']}")
        print(f"Stochastic Expected Revenue: ${run_results_stoch['expected_revenue']:.2f}")
        print("Stochastic DAM Schedule (Stage 1 - Head):")
        # Format DAM schedule for display
        dam_schedule_display = run_results_stoch['schedule_df'].copy()
        for col, fmt in [('DAM_LMP', '{:.2f}'), ('DAM_ChargePower_MW', '{:.3f}'), 
                         ('DAM_DischargePower_MW', '{:.3f}'), ('Avg_RTM_LMP_Scenario', '{:.2f}')]:
             if col in dam_schedule_display.columns:
                  dam_schedule_display[col] = dam_schedule_display[col].map(lambda x: fmt.format(x) if pd.notna(x) else 'NaN')
        print(dam_schedule_display.head().to_string())
        
        # Optionally print scenario details (can be verbose)
        # print("\nSample RTM Scenario 0 Prices (Head):")
        # print(run_results_stoch['rtm_scenario_prices'][0].head())
        
    else:
        print(f"Stochastic Failed: {run_results_stoch['error_message']}")


    print("\nOptimizer command-line tests finished.") 