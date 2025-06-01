import os
import pandas as pd
import pulp
from datetime import datetime, timedelta, timezone
from math import sqrt
from dotenv import load_dotenv
import traceback  # For detailed error logging
import random  # For scenario generation placeholder

# Import the ERCOT API fetcher
from ercot_api_fetcher import fetch_day_ahead_prices, fetch_real_time_lmp, ERCOTAPIError

# Load environment variables (including BESS parameters)
load_dotenv()


# --- Configuration Loading ---
def load_bess_config():
    """Loads BESS configuration from environment variables."""
    try:
        config = {
            "capacity_mwh": float(os.getenv("BESS_CAPACITY_MWH", 10)),
            "max_charge_power_mw": float(os.getenv("BESS_MAX_CHARGE_POWER_MW", 5)),
            "max_discharge_power_mw": float(
                os.getenv("BESS_MAX_DISCHARGE_POWER_MW", 5)
            ),
            "min_soc_percent": float(os.getenv("BESS_MIN_SOC_PERCENT", 10)),
            "max_soc_percent": float(os.getenv("BESS_MAX_SOC_PERCENT", 90)),
            "round_trip_efficiency_percent": float(
                os.getenv("BESS_ROUND_TRIP_EFFICIENCY_PERCENT", 85)
            ),
            "initial_soc_percent": float(os.getenv("BESS_INITIAL_SOC_PERCENT", 50)),
            # Battery degradation parameters
            "battery_cost_usd_per_kwh": float(
                os.getenv("BESS_COST_USD_PER_KWH", 300)
            ),  # Cost of battery in $/kWh
            "cycle_life_at_100pct_dod": float(
                os.getenv("BESS_CYCLE_LIFE_100PCT_DOD", 3000)
            ),  # Number of full cycles before EOL at 100% DoD
            "cycle_life_at_10pct_dod": float(
                os.getenv("BESS_CYCLE_LIFE_10PCT_DOD", 15000)
            ),  # Number of full cycles before EOL at 10% DoD
            "min_cycle_count": float(
                os.getenv("BESS_MIN_CYCLE_COUNT", 0.02)
            ),  # Minimum cycle fraction to count per hour (to prevent excessive small cycles)
        }

        # Convert percentages to decimals
        config["min_soc_mwh"] = (
            config["capacity_mwh"] * config["min_soc_percent"] / 100.0
        )
        config["max_soc_mwh"] = (
            config["capacity_mwh"] * config["max_soc_percent"] / 100.0
        )
        config["initial_soc_mwh"] = (
            config["capacity_mwh"] * config["initial_soc_percent"] / 100.0
        )
        config["charge_efficiency"] = sqrt(
            config["round_trip_efficiency_percent"] / 100.0
        )
        config["discharge_efficiency"] = sqrt(
            config["round_trip_efficiency_percent"] / 100.0
        )

        # Derive battery degradation cost parameters
        # Convert battery cost from $/kWh to $/MWh
        config["battery_cost_usd_per_mwh"] = config["battery_cost_usd_per_kwh"] * 1000

        # Total battery capital cost
        config["total_battery_cost"] = (
            config["capacity_mwh"] * config["battery_cost_usd_per_mwh"]
        )

        # Calculate rainflow cycle counting parameters
        # We use a simple linear model between 10% DoD and 100% DoD for cycle life
        # Marginal degradation cost calculation ($/MWh cycled)
        config["marginal_degradation_cost_100pct"] = (
            config["total_battery_cost"] / config["cycle_life_at_100pct_dod"]
        )
        config["marginal_degradation_cost_10pct"] = (
            config["total_battery_cost"] / config["cycle_life_at_10pct_dod"] / 0.1
        )

        # Slope and intercept for linear degradation cost model
        # cost = m * DoD + b
        m = (
            config["marginal_degradation_cost_100pct"]
            - config["marginal_degradation_cost_10pct"]
        ) / 0.9
        b = config["marginal_degradation_cost_10pct"] - m * 0.1
        config["degradation_cost_slope"] = m
        config["degradation_cost_intercept"] = b

        return config
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Error parsing BESS configuration from .env file: {e}. Ensure all BESS parameters are valid numbers."
        )


# --- Deterministic Optimization Logic ---
def optimize_day_ahead_arbitrage(bess_config: dict, prices_df: pd.DataFrame) -> tuple:
    """
    Optimizes BESS dispatch for day-ahead energy arbitrage using PuLP (Deterministic).
    Assumes perfect foresight based on the input prices_df.
    """
    # --- Model Setup ---
    prob = pulp.LpProblem("BESS_DayAhead_Arbitrage_Deterministic", pulp.LpMaximize)

    # Get the number of intervals from the prices DataFrame
    # If the input data is hourly, we can assume 1 cycle per 24 intervals.
    # If it is in 15 minute intervals then we can assume 1 cycle per 96 intervals
    # Get the time delta of the prices_df to determine the cycle count
    dt = prices_df["HourEnding"].diff().dropna().iloc[0]
    if dt == timedelta(hours=1):
        N = 24  # 24 hours in a day
    elif dt == timedelta(minutes=15):
        N = 96  # 96 intervals in a day (15 minutes each)
    else:
        raise ValueError(
            "Unsupported time interval in prices_df. Expected hourly or 15-minute intervals."
        )

    intervals = range(len(prices_df))

    # --- Decision Variables ---
    charge_power = pulp.LpVariable.dicts(
        "ChargePower",
        intervals,
        lowBound=0,
        upBound=bess_config["max_charge_power_mw"],
        cat="Continuous",
    )
    discharge_power = pulp.LpVariable.dicts(
        "DischargePower",
        intervals,
        lowBound=0,
        upBound=bess_config["max_discharge_power_mw"],
        cat="Continuous",
    )
    soc_mwh = pulp.LpVariable.dicts(
        "SoC",
        range(len(intervals) + 1),
        lowBound=bess_config["min_soc_mwh"],
        upBound=bess_config["max_soc_mwh"],
        cat="Continuous",
    )
    is_charging = pulp.LpVariable.dicts("IsCharging", intervals, cat="Binary")

    # --- Degradation Modeling Variables ---
    # Energy throughput variables (energy cycled through the battery)
    energy_throughput = pulp.LpVariable.dicts(
        "EnergyThroughput", intervals, lowBound=0, cat="Continuous"
    )

    # Variables to track SoC changes for DoD calculation
    soc_decrease = pulp.LpVariable.dicts(
        "SoC_Decrease", intervals, lowBound=0, cat="Continuous"
    )
    max_dod = pulp.LpVariable("Max_DoD", lowBound=0, upBound=1, cat="Continuous")

    # Cycle counting variable (fraction of a cycle in each hour)
    # This is a more sophisticated way to track cycles, but it breaks the linearity of the model,
    # so we will use a simplified version here.
    # cycle_count = pulp.LpVariable.dicts(
    #     "CycleCount", hours, lowBound=0, cat="Continuous"
    # )

    cycle_count = 1.0 / N  # Assume 1 cycle per day for simplicity

    # Degradation cost variable
    degradation_cost = pulp.LpVariable.dicts(
        "DegradationCost", intervals, lowBound=0, cat="Continuous"
    )

    # --- Objective Function ---
    # Original revenue part
    revenue = pulp.lpSum(
        (
            discharge_power[h] * prices_df.loc[h, "LMP"]
            - charge_power[h] * prices_df.loc[h, "LMP"]
        )
        for h in intervals
    )

    # Degradation cost part
    total_degradation_cost = pulp.lpSum(degradation_cost[h] for h in intervals)

    # Net revenue (revenue minus degradation cost)
    prob += revenue - total_degradation_cost, "Net_Revenue"

    # --- Constraints ---
    # Initial SoC
    prob += soc_mwh[0] == bess_config["initial_soc_mwh"], "Initial_SoC"

    discharge_efficiency_inverse = 1.0 / bess_config["discharge_efficiency"]

    # SoC balance and power constraints
    for h in intervals:
        # SoC balance equation
        prob += (
            soc_mwh[h + 1]
            == soc_mwh[h]
            + (charge_power[h] * bess_config["charge_efficiency"])
            - (discharge_power[h] * discharge_efficiency_inverse),
            f"SoC_Balance_Hour_{h}",
        )

        # Prevent simultaneous charge/discharge
        M = (
            max(
                bess_config["max_charge_power_mw"],
                bess_config["max_discharge_power_mw"],
            )
            * 1.1
        )
        prob += charge_power[h] <= M * is_charging[h], f"Charge_Indicator_{h}"
        prob += (
            discharge_power[h] <= M * (1 - is_charging[h]),
            f"Discharge_Indicator_{h}",
        )

        # Degradation modeling constraints
        # Energy throughput calculation (use the smaller efficiency to be conservative)
        min_efficiency = min(
            bess_config["charge_efficiency"], bess_config["discharge_efficiency"]
        )
        prob += (
            energy_throughput[h] == charge_power[h] * min_efficiency,
            f"Energy_Throughput_{h}",
        )

        # Calculate SoC decrease for DoD tracking
        if h > 0:
            prob += soc_decrease[h] >= soc_mwh[h] - soc_mwh[h + 1], f"SoC_Decrease_{h}"

            # Track maximum DoD (as a fraction of usable capacity)
            usable_capacity = bess_config["max_soc_mwh"] - bess_config["min_soc_mwh"]
            prob += (
                max_dod >= soc_decrease[h] / usable_capacity,
                f"Max_DoD_Constraint_{h}",
            )

        # Calculate cycle fraction
        # We use energy throughput method: 1 cycle = using the full capacity once
        # Drop the cycle_count variable for simplicity in this deterministic model
        # prob += (
        #     cycle_count[h] >= energy_throughput[h] / bess_config["capacity_mwh"],
        #     f"Cycle_Count_{h}",
        # )
        # prob += cycle_count[h] >= bess_config["min_cycle_count"], f"Min_Cycle_Count_{h}"

        # Calculate degradation cost based on throughput and DoD
        # Linear degradation cost model: cost = m * DoD + b
        slope = bess_config["degradation_cost_slope"]
        intercept = bess_config["degradation_cost_intercept"]

        # Simplified degradation cost calculation - using current max DoD for all cycles
        # In a more sophisticated model, we would track each cycle's DoD separately

        prob += (
            degradation_cost[h] >= cycle_count * (slope * max_dod + intercept),
            f"Degradation_Cost_{h}",
        )

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # --- Process Results ---
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        schedule = []
        total_revenue = pulp.value(revenue)
        total_degradation = pulp.value(total_degradation_cost)
        net_revenue = pulp.value(prob.objective)

        for h in intervals:
            schedule.append(
                {
                    "HourEnding": prices_df.loc[h, "HourEnding"],
                    "LMP": prices_df.loc[h, "LMP"],
                    "ChargePower_MW": charge_power[h].varValue,
                    "DischargePower_MW": discharge_power[h].varValue,
                    "SoC_MWh_End": soc_mwh[h + 1].varValue,
                    "SoC_Percent_End": (
                        soc_mwh[h + 1].varValue / bess_config["capacity_mwh"]
                    )
                    * 100,
                    "CycleFraction": cycle_count,  # cycle_count[h].varValue if h in cycle_count else 0,
                    "DegradationCost": (
                        degradation_cost[h].varValue if h in degradation_cost else 0
                    ),
                }
            )

        schedule_df = pd.DataFrame(schedule)

        # Add summary data to the results
        result_data = {
            "status": status,
            "schedule_df": schedule_df,
            "total_revenue": total_revenue,
            "degradation_cost": total_degradation,
            "net_revenue": net_revenue,
            "total_cycles": sum(cycle_count for h in intervals),
            "max_dod_percent": max_dod.varValue * 100 if max_dod.varValue else 0,
        }

        return status, schedule_df, net_revenue, result_data
    else:
        print(f"Deterministic optimization failed with status: {status}")
        return status, None, None, None


# --- NEW: Scenario Generation Placeholder ---
def generate_rtm_price_scenarios(
    dam_prices_df: pd.DataFrame, num_scenarios: int, noise_std_dev: float = 5.0
) -> dict:
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
    if "LMP" not in dam_prices_df.columns:
        raise ValueError("DAM prices DataFrame must contain 'LMP' column.")

    for s in range(num_scenarios):
        # Add random noise (mean 0) to DAM prices
        # Ensure prices don't go below a certain floor (e.g., $0 or a typical floor)
        noise = [random.gauss(0, noise_std_dev) for _ in range(len(dam_prices_df))]
        # Use .copy() to avoid modifying the original DAM LMP Series
        rtm_lmp = dam_prices_df["LMP"].copy() + noise
        rtm_lmp = rtm_lmp.clip(lower=0)  # Example: prevent negative prices
        scenarios[s] = rtm_lmp.reset_index(
            drop=True
        )  # Ensure clean index for .loc access later

    print(
        f"Generated {num_scenarios} RTM price scenarios with std dev {noise_std_dev}."
    )
    return scenarios


# --- NEW: Two-Stage Stochastic Optimization Logic with Degradation ---
def optimize_two_stage_stochastic(
    bess_config: dict, dam_prices_df: pd.DataFrame, rtm_scenarios: dict
) -> tuple:
    """
    Optimizes BESS dispatch using a two-stage stochastic model with degradation costs.
    Stage 1: DAM Charge/Discharge decisions (here-and-now).
    Stage 2: RTM Charge/Discharge adjustments & SoC (wait-and-see, per scenario).

    Args:
        bess_config: Dictionary containing BESS parameters.
        dam_prices_df: DataFrame with DAM 'HourEnding' and 'LMP'.
        rtm_scenarios: Dictionary {scenario_index: pd.Series(RTM_LMP)}

    Returns:
        A tuple containing: (status, dam_schedule_df, expected_revenue, result_data)
        status (str): PuLP solver status.
        dam_schedule_df (pd.DataFrame or None): DataFrame with optimal Stage 1 (DAM) schedule.
        expected_revenue (float or None): Calculated maximum expected revenue across scenarios.
        result_data (dict): Additional result data including degradation costs.
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
    dam_charge_power = pulp.LpVariable.dicts(
        "DAM_ChargePower",
        hours,
        lowBound=0,
        upBound=bess_config["max_charge_power_mw"],
        cat="Continuous",
    )
    dam_discharge_power = pulp.LpVariable.dicts(
        "DAM_DischargePower",
        hours,
        lowBound=0,
        upBound=bess_config["max_discharge_power_mw"],
        cat="Continuous",
    )
    # Binary DAM variable to prevent simultaneous charge/discharge
    is_dam_charging = pulp.LpVariable.dicts("Is_DAM_Charging", hours, cat="Binary")

    # Stage 2 (RTM Adjustments & SoC - per scenario)
    # Using tuples (hour, scenario) as keys
    rtm_charge_power = pulp.LpVariable.dicts(
        "RTM_ChargePower",
        [(h, s) for h in hours for s in scenarios_idx],
        lowBound=0,
        cat="Continuous",
    )
    rtm_discharge_power = pulp.LpVariable.dicts(
        "RTM_DischargePower",
        [(h, s) for h in hours for s in scenarios_idx],
        lowBound=0,
        cat="Continuous",
    )
    # SoC exists for hour 0 through final hour, per scenario
    soc_mwh = pulp.LpVariable.dicts(
        "SoC",
        [(t, s) for t in range(len(hours) + 1) for s in scenarios_idx],
        lowBound=bess_config["min_soc_mwh"],
        upBound=bess_config["max_soc_mwh"],
        cat="Continuous",
    )

    # --- Degradation Modeling Variables ---
    # Total energy throughput variables (per hour, per scenario)
    energy_throughput = pulp.LpVariable.dicts(
        "EnergyThroughput",
        [(h, s) for h in hours for s in scenarios_idx],
        lowBound=0,
        cat="Continuous",
    )

    # Variables to track SoC changes for DoD calculation (per hour, per scenario)
    soc_decrease = pulp.LpVariable.dicts(
        "SoC_Decrease",
        [(h, s) for h in hours for s in scenarios_idx],
        lowBound=0,
        cat="Continuous",
    )

    # Maximum DoD per scenario
    max_dod = pulp.LpVariable.dicts(
        "Max_DoD", scenarios_idx, lowBound=0, upBound=1, cat="Continuous"
    )

    # Expected maximum DoD across all scenarios
    expected_max_dod = pulp.LpVariable(
        "Expected_Max_DoD", lowBound=0, upBound=1, cat="Continuous"
    )

    # Cycle counting variable (per hour, per scenario)
    # cycle_count = pulp.LpVariable.dicts(
    #     "CycleCount",
    #     [(h, s) for h in hours for s in scenarios_idx],
    #     lowBound=0,
    #     cat="Continuous",
    # )
    cycle_count = 1.0 / 24.0  # Assume 1 cycle per day for simplicity

    # Degradation cost variable (per hour, per scenario)
    degradation_cost = pulp.LpVariable.dicts(
        "DegradationCost",
        [(h, s) for h in hours for s in scenarios_idx],
        lowBound=0,
        cat="Continuous",
    )

    # --- Objective Function (Maximize Expected Net Profit) ---
    # DAM Revenue
    dam_revenue = pulp.lpSum(
        (dam_discharge_power[h] - dam_charge_power[h]) * dam_prices_df.loc[h, "LMP"]
        for h in hours
    )

    # Expected RTM Revenue
    expected_rtm_revenue = pulp.lpSum(
        scenario_probability
        * (rtm_discharge_power[h, s] - rtm_charge_power[h, s])
        * rtm_scenarios[s].iloc[h]
        for h in hours
        for s in scenarios_idx
    )

    # Expected Degradation Cost
    expected_degradation_cost = pulp.lpSum(
        scenario_probability * degradation_cost[h, s]
        for h in hours
        for s in scenarios_idx
    )

    # Total Expected Net Profit
    prob += (
        dam_revenue + expected_rtm_revenue - expected_degradation_cost,
        "Total_Expected_Net_Revenue",
    )

    # --- Constraints ---
    # ** Stage 1 Constraints **
    # Prevent Simultaneous DAM Charge and Discharge
    M_dam = (
        max(bess_config["max_charge_power_mw"], bess_config["max_discharge_power_mw"])
        * 1.1
    )  # Big-M
    for h in hours:
        prob += (
            dam_charge_power[h] <= M_dam * is_dam_charging[h],
            f"DAM_Charge_Indicator_{h}",
        )
        prob += (
            dam_discharge_power[h] <= M_dam * (1 - is_dam_charging[h]),
            f"DAM_Discharge_Indicator_{h}",
        )

    # ** Stage 2 Constraints (Applied PER SCENARIO) **
    discharge_efficiency_inverse = 1.0 / bess_config["discharge_efficiency"]
    charge_efficiency = bess_config["charge_efficiency"]
    min_efficiency = min(charge_efficiency, bess_config["discharge_efficiency"])

    # Expected max DoD calculation
    prob += (
        expected_max_dod
        == pulp.lpSum(scenario_probability * max_dod[s] for s in scenarios_idx),
        "Expected_Max_DoD_Calculation",
    )

    for s in scenarios_idx:
        # Initial SoC (same for all scenarios)
        prob += (
            soc_mwh[(0, s)] == bess_config["initial_soc_mwh"],
            f"Initial_SoC_Scenario_{s}",
        )

        for h in hours:
            # Total power in the hour (DAM + RTM adjustment)
            total_charge = dam_charge_power[h] + rtm_charge_power[h, s]
            total_discharge = dam_discharge_power[h] + rtm_discharge_power[h, s]

            # SoC Balance for this scenario
            # Ensure correct tuple indexing for SoC
            prob += (
                soc_mwh[(h + 1, s)]
                == soc_mwh[(h, s)]
                + (total_charge * charge_efficiency)
                - (total_discharge * discharge_efficiency_inverse),
                f"SoC_Balance_Hour_{h}_Scenario_{s}",
            )

            # Total Power Limits for this scenario
            prob += (
                total_charge <= bess_config["max_charge_power_mw"],
                f"MaxChargeLimit_Hour_{h}_Scenario_{s}",
            )
            prob += (
                total_discharge <= bess_config["max_discharge_power_mw"],
                f"MaxDischargeLimit_Hour_{h}_Scenario_{s}",
            )

            # Energy throughput calculation
            prob += (
                energy_throughput[h, s] == total_charge * min_efficiency,
                f"Energy_Throughput_Hour_{h}_Scenario_{s}",
            )

            # Calculate SoC decrease for DoD tracking
            if h > 0:
                prob += (
                    soc_decrease[h, s] >= soc_mwh[(h, s)] - soc_mwh[(h + 1, s)],
                    f"SoC_Decrease_Hour_{h}_Scenario_{s}",
                )

                # Track maximum DoD (as a fraction of usable capacity)
                usable_capacity = (
                    bess_config["max_soc_mwh"] - bess_config["min_soc_mwh"]
                )
                prob += (
                    max_dod[s] >= soc_decrease[h, s] / usable_capacity,
                    f"Max_DoD_Constraint_Hour_{h}_Scenario_{s}",
                )

            # We simplified this to a constant cycle count for simplicity
            # # Calculate cycle fraction
            # prob += (
            #     cycle_count[h, s]
            #     >= energy_throughput[h, s] / bess_config["capacity_mwh"],
            #     f"Cycle_Count_Hour_{h}_Scenario_{s}",
            # )
            # prob += (
            #     cycle_count[h, s] >= bess_config["min_cycle_count"],
            #     f"Min_Cycle_Count_Hour_{h}_Scenario_{s}",
            # )

            # Calculate degradation cost based on throughput and DoD
            # Linear degradation cost model: cost = m * DoD + b
            slope = bess_config["degradation_cost_slope"]
            intercept = bess_config["degradation_cost_intercept"]

            # Simplified degradation cost calculation - using current max DoD for this scenario
            prob += (
                degradation_cost[h, s]
                >= cycle_count * (slope * max_dod[s] + intercept),
                f"Degradation_Cost_Hour_{h}_Scenario_{s}",
            )

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # --- Process Results ---
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        # Extract the Stage 1 (DAM) schedule - this is the primary output
        dam_schedule = []

        # Get objective components
        dam_revenue_value = pulp.lpSum(
            (dam_discharge_power[h].varValue - dam_charge_power[h].varValue)
            * dam_prices_df.loc[h, "LMP"]
            for h in hours
        ).value()

        expected_rtm_revenue_value = pulp.lpSum(
            scenario_probability
            * (rtm_discharge_power[h, s].varValue - rtm_charge_power[h, s].varValue)
            * rtm_scenarios[s].iloc[h]
            for h in hours
            for s in scenarios_idx
        ).value()

        expected_degradation_cost_value = pulp.lpSum(
            scenario_probability * degradation_cost[h, s].varValue
            for h in hours
            for s in scenarios_idx
        ).value()

        expected_net_revenue = pulp.value(prob.objective)

        # Calculate average cycle count per scenario
        avg_cycle_count = sum(
            scenario_probability * sum(cycle_count for h in hours)
            for s in scenarios_idx
        )

        # Calculate expected max DoD
        expected_max_dod_value = pulp.value(expected_max_dod)

        # Build schedule dataframe
        for h in hours:
            dam_schedule.append(
                {
                    "HourEnding": dam_prices_df.loc[h, "HourEnding"],
                    "DAM_LMP": dam_prices_df.loc[
                        h, "LMP"
                    ],  # Include DAM price for context
                    "DAM_ChargePower_MW": dam_charge_power[h].varValue,
                    "DAM_DischargePower_MW": dam_discharge_power[h].varValue,
                    # Add RTM price stats for context
                    "Avg_RTM_LMP_Scenario": sum(
                        rtm_scenarios[s].iloc[h] for s in scenarios_idx
                    )
                    / num_scenarios,
                    # Add average degradation cost across scenarios for this hour
                    "Avg_Degradation_Cost": sum(
                        scenario_probability * degradation_cost[h, s].varValue
                        for s in scenarios_idx
                    ),
                }
            )

        dam_schedule_df = pd.DataFrame(dam_schedule)

        # Prepare additional result data
        result_data = {
            "dam_revenue": dam_revenue_value,
            "expected_rtm_revenue": expected_rtm_revenue_value,
            "expected_degradation_cost": expected_degradation_cost_value,
            "expected_revenue": expected_net_revenue,
            "avg_cycle_count": avg_cycle_count,
            "expected_max_dod_percent": (
                expected_max_dod_value * 100 if expected_max_dod_value else 0
            ),
            "rtm_scenario_prices": rtm_scenarios,
        }

        return status, dam_schedule_df, expected_net_revenue, result_data
    else:
        print(f"Stochastic optimization failed with status: {status}")
        return status, None, None, None


# --- Data Fetching and Processing Function for Real Time 15 minute market ---
def fetch_and_process_rtm_prices(
    settlement_point: str,
    start_time: datetime = datetime.now(timezone.utc),
) -> pd.DataFrame:
    """
    Fetches and processes Day-Ahead Market prices for the given date and settlement point.
    Returns a cleaned DataFrame with 'HourEnding' (datetime) and 'LMP' (float).
    Raises ERCOTAPIError or ValueError on failure.
    """
    try:
        print(
            f"Fetching Real Time prices for latest 15 minutes at {settlement_point}..."
        )
        raw_prices = fetch_real_time_lmp(
            settlement_point=settlement_point, start_time=start_time
        )
        # Sort raw_prices by date:
        raw_prices = sorted(raw_prices, key=lambda x: (x[0], x[1], x[2]))

        start_date = datetime.strptime(
            raw_prices[0][0],
            "%Y-%m-%d",
        )  # Assuming first entry has the start date

        # Process raw prices into a DataFrame
        processed_prices = []
        for i, p in enumerate(raw_prices):
            # Create proper datetime for HourEnding
            hour_ending = datetime.combine(start_date, datetime.min.time()) + timedelta(
                minutes=(i + 1) * 15
            )
            processed_prices.append({"HourEnding": hour_ending, "LMP": float(p[5])})

        df = pd.DataFrame(processed_prices)
        return df

    except ERCOTAPIError as e:
        raise ERCOTAPIError(f"Failed to fetch DAM prices: {e}")
    except Exception as e:
        raise ValueError(f"Error processing DAM prices: {e}")


# --- Data Fetching and Processing Function ---
def fetch_and_process_dam_prices(
    settlement_point: str, target_date: datetime.date
) -> pd.DataFrame:
    """
    Fetches and processes Day-Ahead Market prices for the given date and settlement point.
    Returns a cleaned DataFrame with 'HourEnding' (datetime) and 'LMP' (float).
    Raises ERCOTAPIError or ValueError on failure.
    """
    try:
        print(f"Fetching Day-Ahead prices for {target_date} at {settlement_point}...")
        raw_prices = fetch_day_ahead_prices(
            settlement_point=settlement_point, delivery_date=target_date
        )

        # Ensure data for 24 hours
        if len(raw_prices) != 24:
            print(f"Warning: Expected 24 hourly prices, but got {len(raw_prices)}")

        # Process raw prices into a DataFrame
        processed_prices = []
        for i, p in enumerate(raw_prices):
            # Create proper datetime for HourEnding
            hour_ending = datetime.combine(
                target_date, datetime.min.time()
            ) + timedelta(hours=i + 1)
            processed_prices.append({"HourEnding": hour_ending, "LMP": float(p[3])})

        df = pd.DataFrame(processed_prices)
        print(f"Successfully fetched and processed {len(df)} hourly prices.")
        return df

    except ERCOTAPIError as e:
        raise ERCOTAPIError(f"Failed to fetch DAM prices: {e}")
    except Exception as e:
        raise ValueError(f"Error processing DAM prices: {e}")


# --- Main Pipeline Function to Orchestrate the Process ---
def run_optimization_pipeline(
    target_date: datetime.date,
    settlement_point: str,
    optimization_type: str = "deterministic",  # 'deterministic' or 'stochastic'
    num_scenarios: int = 5,  # Used only if type is stochastic
    noise_std_dev: float = 5.0,  # Used only if type is stochastic
    market_type: str = "RTM",  # 'DAM' or 'RTM'
) -> dict:
    """
    Orchestrates the full BESS optimization pipeline, from data fetching to result processing.

    Args:
        target_date: The operating date to optimize for.
        settlement_point: The ERCOT settlement point to fetch prices for.
        optimization_type: 'deterministic' for perfect foresight, 'stochastic' for two-stage stochastic.
        num_scenarios: Number of price scenarios to generate (stochastic only).
        noise_std_dev: Standard deviation of price noise for scenario generation (stochastic only).

    Returns:
        A dictionary containing results and optimization outputs.
    """
    try:
        # Standardize parameters
        target_date = pd.to_datetime(target_date).date()
        settlement_point = settlement_point.strip().upper()
        optimization_type = optimization_type.lower()

        # Load BESS configuration
        bess_config = load_bess_config()

        if market_type.lower() == "dam":
            # Fetch & process DAM prices
            prices_df = fetch_and_process_dam_prices(settlement_point, target_date)
        elif market_type.lower() == "rtm":
            # Fetch & process RTM prices
            prices_df = fetch_and_process_rtm_prices(settlement_point)
        else:
            raise ValueError(
                f"Invalid market_type: {market_type}. Must be 'DAM' or 'RTM'."
            )

        # Perform optimization based on type
        if optimization_type == "deterministic":
            print("Running Deterministic Optimization...")
            status, schedule_df, total_revenue, result_data = (
                optimize_day_ahead_arbitrage(bess_config, prices_df)
            )

            results = {
                "success": status == "Optimal",
                "status": status,
                "schedule_df": schedule_df,
                "prices_df": prices_df,
                "total_revenue": total_revenue,
                "error_message": (
                    None
                    if status == "Optimal"
                    else "Optimization failed to find optimal solution."
                ),
            }

            # Add degradation data if available
            if result_data:
                results.update(
                    {
                        "degradation_cost": result_data.get("degradation_cost", 0),
                        "net_revenue": result_data.get("net_revenue", 0),
                        "total_cycles": result_data.get("total_cycles", 0),
                        "max_dod_percent": result_data.get("max_dod_percent", 0),
                    }
                )

        elif optimization_type == "stochastic":
            print(
                f"Running Two-Stage Stochastic Optimization with {num_scenarios} scenarios..."
            )
            # Generate RTM scenarios
            rtm_scenarios = generate_rtm_price_scenarios(
                prices_df, num_scenarios, noise_std_dev
            )

            # Run stochastic optimization
            status, schedule_df, expected_revenue, result_data = (
                optimize_two_stage_stochastic(bess_config, prices_df, rtm_scenarios)
            )

            results = {
                "success": status == "Optimal",
                "status": status,
                "schedule_df": schedule_df,
                "prices_df": prices_df,
                "expected_revenue": expected_revenue,
                "rtm_scenario_prices": rtm_scenarios if status == "Optimal" else None,
                "error_message": (
                    None
                    if status == "Optimal"
                    else "Optimization failed to find optimal solution."
                ),
            }

            # Add degradation data if available
            if result_data:
                results.update(
                    {
                        "dam_revenue": result_data.get("dam_revenue", 0),
                        "expected_rtm_revenue": result_data.get(
                            "expected_rtm_revenue", 0
                        ),
                        "expected_degradation_cost": result_data.get(
                            "expected_degradation_cost", 0
                        ),
                        "avg_cycle_count": result_data.get("avg_cycle_count", 0),
                        "expected_max_dod_percent": result_data.get(
                            "expected_max_dod_percent", 0
                        ),
                    }
                )

        else:
            raise ValueError(
                f"Invalid optimization_type: {optimization_type}. Must be 'deterministic' or 'stochastic'."
            )

        return results

    except Exception as e:
        print(f"Pipeline Error: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "status": "Error",
            "error_message": str(e),
            "schedule_df": None,
            "prices_df": None,
        }


# --- Main Execution Block (UPDATED for testing both modes) ---
if __name__ == "__main__":
    print("Running BESS Optimizer (Command Line)...")

    # Example usage parameters
    target_date_cli = datetime.now(timezone.utc).date() - timedelta(days=1)
    settlement_point_cli = "HB_NORTH"

    # --- Test Deterministic ---
    print("\n--- TESTING DETERMINISTIC ---")
    run_results_det = run_optimization_pipeline(
        target_date_cli, settlement_point_cli, optimization_type="deterministic"
    )
    if run_results_det["success"]:
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
        optimization_type="stochastic",
        num_scenarios=5,  # Example: 5 scenarios
        noise_std_dev=10.0,  # Example: +/- $10/MWh noise std dev
    )
    if run_results_stoch["success"]:
        print(f"Stochastic Status: {run_results_stoch['status']}")
        print(
            f"Stochastic Expected Revenue: ${run_results_stoch['expected_revenue']:.2f}"
        )
        print("Stochastic DAM Schedule (Stage 1 - Head):")
        # Format DAM schedule for display
        dam_schedule_display = run_results_stoch["schedule_df"].copy()
        for col, fmt in [
            ("DAM_LMP", "{:.2f}"),
            ("DAM_ChargePower_MW", "{:.3f}"),
            ("DAM_DischargePower_MW", "{:.3f}"),
            ("Avg_RTM_LMP_Scenario", "{:.2f}"),
        ]:
            if col in dam_schedule_display.columns:
                dam_schedule_display[col] = dam_schedule_display[col].map(
                    lambda x: fmt.format(x) if pd.notna(x) else "NaN"
                )
        print(dam_schedule_display.head().to_string())

        # Optionally print scenario details (can be verbose)
        # print("\nSample RTM Scenario 0 Prices (Head):")
        # print(run_results_stoch['rtm_scenario_prices'][0].head())

    else:
        print(f"Stochastic Failed: {run_results_stoch['error_message']}")

    print("\nOptimizer command-line tests finished.")
