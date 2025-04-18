import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# --- Streamlit Page Configuration ---
# This must be the first Streamlit command
st.set_page_config(layout="wide", page_title="BESS Optimization Dashboard")

# Import the refactored optimizer functions
# Note: Ensure bess_optimizer.py has been updated for stochastic mode
from bess_optimizer import run_optimization_pipeline, load_bess_config

# --- Load and apply CSS ---
def load_css(css_file):
    with open(css_file, 'r') as f:
        css = f.read()
    return css

css = load_css('style.css')
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# --- Helper Functions ---
def format_schedule_display(df, opt_type):
    """Formats the schedule DataFrame for better display in Streamlit."""
    if df is None:
        return None
    display_df = df.copy()
    
    # Define columns and formats based on optimization type
    if opt_type == 'stochastic':
        cols_formats = [
            ('DAM_LMP', '{:.2f}'), 
            ('Avg_RTM_LMP_Scenario', '{:.2f}'),
            ('DAM_ChargePower_MW', '{:.3f}'), 
            ('DAM_DischargePower_MW', '{:.3f}')
            # SoC columns are not directly available for DAM schedule in stochastic
        ]
    else: # Deterministic
        cols_formats = [
            ('LMP', '{:.2f}'), 
            ('ChargePower_MW', '{:.3f}'), 
            ('DischargePower_MW', '{:.3f}'), 
            ('SoC_MWh_End', '{:.3f}'), 
            ('SoC_Percent_End', '{:.1f}')
        ]
        
    # Format numeric columns
    for col, fmt in cols_formats:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: fmt.format(x) if pd.notna(x) else 'NaN')
            
    # Format datetime
    if 'HourEnding' in display_df.columns:
        display_df['HourEnding'] = display_df['HourEnding'].dt.strftime('%Y-%m-%d %H:%M')
    return display_df

def create_combined_chart(schedule_df, prices_df, opt_type):
    """
    Creates a single Plotly figure with two subplots (Price and Schedule) 
    sharing the x-axis. Adapts based on optimization type.
    UPDATED: Adjusts legend for better mobile view.
    """
    is_stochastic = (opt_type == 'stochastic')
    
    # Check required dataframes
    if prices_df is None or prices_df.empty or schedule_df is None or schedule_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.update_layout(title_text="Price and Dispatch Schedule (Data Missing)", height=600)
        return fig
        
    # Determine columns based on type
    price_col = 'DAM_LMP' if is_stochastic else 'LMP'
    charge_col = 'DAM_ChargePower_MW' if is_stochastic else 'ChargePower_MW'
    discharge_col = 'DAM_DischargePower_MW' if is_stochastic else 'DischargePower_MW'
    
    # Determine SoC column or calculate it if needed
    if not is_stochastic:
        soc_col = 'SoC_Percent_End'  # Already available for deterministic
    else:
        # For stochastic, check if we have SoC data
        if 'SoC_Percent_End' in schedule_df.columns:
            soc_col = 'SoC_Percent_End'
        else:
            # Calculate SoC from charge/discharge if not available directly
            # Get BESS config for capacity and efficiency
            try:
                bess_config = load_bess_config()
                capacity_mwh = bess_config["capacity_mwh"]
                initial_soc_percent = bess_config["initial_soc_percent"]
                charge_efficiency = bess_config["charge_efficiency"]
                discharge_efficiency_inverse = 1.0 / bess_config["discharge_efficiency"]
                
                # Calculate SoC in MWh first
                schedule_df['SoC_MWh_End'] = 0.0
                current_soc_mwh = capacity_mwh * initial_soc_percent / 100.0  # Initial SoC in MWh
                
                for idx, row in schedule_df.iterrows():
                    # Update SoC based on charge/discharge
                    charge_mwh = row[charge_col] * charge_efficiency
                    discharge_mwh = row[discharge_col] * discharge_efficiency_inverse
                    current_soc_mwh = current_soc_mwh + charge_mwh - discharge_mwh
                    schedule_df.at[idx, 'SoC_MWh_End'] = current_soc_mwh
                
                # Convert to percentage
                schedule_df['SoC_Percent_End'] = (schedule_df['SoC_MWh_End'] / capacity_mwh) * 100
                soc_col = 'SoC_Percent_End'
            except Exception as e:
                st.warning(f"Could not calculate State of Charge: {e}")
                soc_col = None
    
    # Check if essential columns exist in schedule_df
    required_sched_cols = [charge_col, discharge_col]
    if soc_col:
        required_sched_cols.append(soc_col)
    
    missing_cols = [col for col in required_sched_cols if col not in schedule_df.columns]
    if missing_cols:
        st.warning(f"Schedule DataFrame missing required columns for charting: {missing_cols}")
        if soc_col in missing_cols:
            # We can still proceed without SoC
            required_sched_cols.remove(soc_col)
            soc_col = None
            
    if not all(col in schedule_df.columns for col in required_sched_cols):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.update_layout(title_text="Price and Dispatch Schedule (Schedule Data Incomplete)", height=600)
        return fig
    
    # Create subplots - always use secondary y-axis if we have SoC
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.3, 0.7], 
        specs=[[{"secondary_y": False}], # Row 1 (Price)
               [{"secondary_y": soc_col is not None}]] # Row 2 (Schedule), add secondary axis if SoC is available
    )

    # --- Subplot 1: Price (DAM LMP) ---
    fig.add_trace(
        go.Scatter(x=prices_df['HourEnding'], y=prices_df['LMP'], name='DAM LMP ($/MWh)', 
                   mode='lines', line=dict(color='purple')),
        row=1, col=1
    )
    # Optionally add Avg RTM LMP from stochastic results to price plot
    if is_stochastic and 'Avg_RTM_LMP_Scenario' in schedule_df.columns:
         fig.add_trace(
            go.Scatter(x=schedule_df['HourEnding'], y=schedule_df['Avg_RTM_LMP_Scenario'], 
                       name='Avg RTM LMP (Scenario)', mode='lines', line=dict(color='orange', dash='dot')),
            row=1, col=1
        )

    # --- Subplot 2: Schedule ---
    # Discharge Power
    fig.add_trace(
        go.Bar(x=schedule_df['HourEnding'], y=schedule_df[discharge_col], name='Discharge MW', 
               marker_color='red'),
        row=2, col=1, secondary_y=False
    )
    # Charge Power
    fig.add_trace(
        go.Bar(x=schedule_df['HourEnding'], y=-schedule_df[charge_col], name='Charge MW', 
               marker_color='green'),
        row=2, col=1, secondary_y=False
    )
    # State of Charge (%) - Add for both deterministic and stochastic if available
    if soc_col:
        fig.add_trace(
            go.Scatter(x=schedule_df['HourEnding'], y=schedule_df[soc_col], name='SoC %', 
                       mode='lines+markers', line=dict(color='blue')),
            row=2, col=1, secondary_y=True
        )

    # --- Layout Updates ---
    chart_title = "Day-Ahead Price and Optimal Dispatch Schedule"
    if is_stochastic:
         chart_title += " (Stochastic DAM Schedule)"
    fig.update_layout(
        title_text=chart_title,
        height=700,
        barmode='relative',
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom",
            y=-0.3, # Adjusted y-position further down
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=120) # Increased bottom margin
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="LMP ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1, secondary_y=False)
    
    # Add SoC y-axis label if we have SoC data
    if soc_col:
        fig.update_yaxes(title_text="SoC (%)", row=2, col=1, secondary_y=True, range=[0, 100], showgrid=False)
    
    fig.update_xaxes(title_text="Hour Ending", row=2, col=1)

    return fig

def render_revenue_explanation():
    """Renders the explanation of how revenue is calculated in the stochastic model."""
    st.markdown("""
    <div class="explanation-box">
        <h4>Understanding Revenue in Stochastic Optimization</h4>
        <p>The <b>Optimal Expected Revenue</b> shown in the dashboard is calculated using a two-stage stochastic model:</p>
        <ol>
            <li><b>Stage 1 (Day-Ahead Market)</b>: The chart below shows only the Day-Ahead Market (DAM) schedule - when to charge/discharge in the DAM. This is the "here-and-now" decision.</li>
            <li><b>Stage 2 (Real-Time Market)</b>: The model considers multiple possible RTM price scenarios and their optimal responses, but these are not shown in the DAM chart.</li>
        </ol>
        <p>Even if the DAM schedule only shows charging (buying energy), the expected revenue accounts for anticipated profits from discharging (selling energy) in the RTM the next day.</p>
        <p>Think of it as <b>strategic positioning</b> - charging at relatively low DAM prices to take advantage of expected higher RTM prices later.</p>
    </div>
    """, unsafe_allow_html=True)

def render_bess_guide_tab():
    """Renders the content for the BESS Optimization Guide tab."""
    st.header("Battery Energy Storage System (BESS) Optimization Guide")
    
    st.subheader("What We're Optimizing")
    st.markdown("""
    This application optimizes the **bidding and operational strategy** for a Battery Energy Storage System (BESS) 
    participating in the ERCOT electricity market. The core objective is to **maximize NET revenue** by determining the 
    optimal times to charge (buy) and discharge (sell) energy, considering both Day-Ahead Market (DAM) and 
    Real-Time Market (RTM) price variations, while accounting for the **cost of battery degradation**.
    """)
    
    st.subheader("Why It Matters")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="explanation-box">
            <h4>Economic Value</h4>
            <p>Effective BESS bidding strategies can significantly increase revenue and ROI for battery projects. 
            In markets with high price volatility like ERCOT, the difference between optimal and sub-optimal 
            strategies can represent millions in annual revenue.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="explanation-box">
            <h4>Grid Stability</h4>
            <p>Optimized BESS operations support grid reliability by storing energy during low-demand periods 
            and providing power during peak demand. This helps integrate renewables, reduce peak prices, and 
            prevent outages during extreme weather events.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Our Optimization Approach")
    
    st.markdown("""
    <div class="explanation-box">
        <h4>Two-Stage Stochastic Optimization</h4>
        <p>Our approach differentiates between Day-Ahead decisions (which must be made in advance) and Real-Time adjustments 
        (which can respond to actual price conditions):</p>
        <ul>
            <li><b>Stage 1 (Day-Ahead Market)</b>: Determine optimal charge/discharge schedule for the DAM based on forecasted 
            prices and considering future uncertainty</li>
            <li><b>Stage 2 (Real-Time Market)</b>: Model potential RTM price scenarios and optimize adjustments for each scenario</li>
        </ul>
        <p>By considering multiple RTM price scenarios, the model finds a robust DAM strategy that performs well 
        under various possible future price conditions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Deterministic Optimization</div>
            <div class="metric-value">Perfect Foresight</div>
            <p>Assumes perfect knowledge of future prices and optimizes accordingly. Useful as a theoretical upper bound on performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Stochastic Optimization</div>
            <div class="metric-value">Price Uncertainty</div>
            <p>Accounts for uncertainty in real-time prices using multiple scenarios. More realistic representation of actual market conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add new section on technical formulation with corrected formatting
    st.subheader("Mathematical Formulation")
    
    st.markdown("### Deterministic Model (Perfect Foresight)")
    
    st.markdown("**Decision Variables:**")
    st.markdown("""
    * Charge Power (MW) for each hour h: `charge_power[h] ≥ 0`
    * Discharge Power (MW) for each hour h: `discharge_power[h] ≥ 0`
    * State of Charge (MWh) at end of each hour h: `soc[h]`
    * Binary variable indicating charging mode: `is_charging[h] ∈ {0,1}`
    * Energy Throughput (MWh) in hour h: `energy_throughput[h]`
    * Degradation Cost ($) for hour h: `degradation_cost[h]`
    """)
    
    st.markdown("**Objective Function:**")
    st.markdown("Maximize Net Revenue: `∑(h) [ (discharge_power[h] - charge_power[h]) × LMP[h] ] - ∑(h) degradation_cost[h]`")
    st.markdown("This maximizes gross arbitrage revenue minus the estimated cost of battery degradation.")
    
    st.markdown("**Key Constraints:**")
    st.markdown("""
    * **Initial SoC:** `soc[0] = initial_soc_mwh`
    * **SoC Evolution:** `soc[h+1] = soc[h] + (charge_power[h] × charge_efficiency) - (discharge_power[h] / discharge_efficiency)`
    * **SoC Limits:** `min_soc_mwh ≤ soc[h] ≤ max_soc_mwh`
    * **Power Limits:** `charge_power[h] ≤ max_charge_power_mw` and `discharge_power[h] ≤ max_discharge_power_mw`
    * **No Simultaneous Charging/Discharging:** Uses binary variable `is_charging[h]`.
    * **Energy Throughput:** `energy_throughput[h] == charge_power[h]` (Approximation based on charge)
    * **Degradation Cost:** `degradation_cost[h] >= energy_throughput[h] * avg_degradation_cost_per_mwh` (Uses simplified avg cost)
    """)
    
    st.markdown("---")
    
    st.markdown("### Stochastic Model (Two-Stage)")
    
    st.markdown("**Decision Variables:**")
    st.markdown("""
    * **Stage 1 (DAM):** `dam_charge_power[h], dam_discharge_power[h]`
    * **Stage 2 (RTM):** `rtm_charge_power[h,s], rtm_discharge_power[h,s]`
    * **State of Charge:** `soc[h,s]`
    * **Energy Throughput:** `energy_throughput[h,s]`
    * **Degradation Cost:** `degradation_cost[h,s]`
    """)
    
    st.markdown("**Objective Function:**")
    st.markdown("Maximize Expected Net Revenue:")
    st.code("""
    Maximize:
      ∑(h) [ (dam_discharge_power[h] - dam_charge_power[h]) × DAM_LMP[h] ] 
      + ∑(s) probability[s] × ∑(h) [ (rtm_discharge_power[h,s] - rtm_charge_power[h,s]) × RTM_LMP[h,s] ]
      - ∑(s) probability[s] × ∑(h) [ degradation_cost[h,s] ]
    """, language="text")
    st.markdown("Maximizes DAM revenue plus expected RTM revenue, minus expected degradation cost.")
    
    st.markdown("**Key Constraints:**")
    st.markdown("""
    * **Initial SoC:** `soc[0,s] = initial_soc_mwh` (for all scenarios s)
    * **SoC Evolution:** 
      `soc[h+1,s] = soc[h,s] + (total_charge[h,s] × charge_efficiency) - (total_discharge[h,s] / discharge_efficiency)`
    * **SoC Limits:** `min_soc_mwh ≤ soc[h,s] ≤ max_soc_mwh`
    * **Power Limits:** Combined DAM + RTM power respects limits.
    * **Non-Anticipativity:** Stage 1 decisions are the same across scenarios.
    * **Energy Throughput:** `energy_throughput[h,s] == total_charge[h,s]` (Approximation)
    * **Degradation Cost:** `degradation_cost[h,s] >= energy_throughput[h,s] * avg_degradation_cost_per_mwh`
    """)
    
    st.markdown("---")
    
    st.markdown("### Interpreting Optimization Results")
    
    st.markdown("**Deterministic Results:**")
    st.markdown("""
    * The optimal schedule shows when to charge (green bars) and discharge (red bars).
    * The **Optimal Net Revenue** metric shows the profit after accounting for estimated degradation costs.
    * State of Charge (%) line shows battery energy level.
    """)
    
    st.markdown("**Stochastic Results:**")
    st.markdown("""
    * The chart shows only the Stage 1 (DAM) decisions.
    * The **Optimal Expected Revenue** metric shows the *expected* profit across all scenarios, after accounting for *expected* degradation costs.
    * The DAM schedule represents a robust strategy considering future uncertainty and degradation.
    """)
    
    st.markdown("""
    **Key Insight:** The optimization now balances immediate arbitrage profit against the long-term cost of battery degradation, aiming for the highest sustainable net revenue.
    """)
    
    st.subheader("How It Works")
    
    st.markdown("""
    <div class="explanation-box">
        <h4>1. Data Acquisition</h4>
        <p>Fetches Day-Ahead Market (DAM) prices from ERCOT API for the selected settlement point and date.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>2. Scenario Generation</h4>
        <p>Creates multiple Real-Time Market (RTM) price scenarios by adding random noise to DAM prices. 
        These scenarios represent the uncertainty in RTM prices that will materialize the next day.</p>
        <p>The noise level can be adjusted to model different degrees of price volatility in the market.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>3. Mathematical Optimization</h4>
        <p>Formulates a linear programming problem using PuLP to maximize **net revenue** (deterministic) or **expected net revenue** (stochastic):</p>
        <ul>
            <li><b>Decision Variables</b>: Charge/discharge power, SoC (plus RTM adjustments in stochastic).</li>
            <li><b>Constraints</b>: Battery limits, energy balance, degradation cost calculation.</li>
            <li><b>Objective</b>: Maximize profit from energy sales minus costs (energy purchase + degradation).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>4. Result Interpretation</h4>
        <p>The results show the optimal schedule and the calculated revenue, explicitly accounting for the estimated cost of battery degradation based on the provided parameters.</p>
        <p>Negative net revenue suggests the estimated degradation cost and efficiency losses outweigh potential arbitrage profits for the selected day and parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Technical Architecture")
    
    st.markdown("""
    * **bess_optimizer.py**: Core optimization logic, scenario generation, and API integration
    * **ercot_api_fetcher.py**: Functions to fetch and process ERCOT market data
    * **streamlit_app.py**: Interactive web dashboard for visualization and parameter settings
    
    <div style="margin-top: 20px; padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid #007aff;">
        <p><b>👨‍💻 Open Source Project:</b> View the code, contribute, or fork this project on GitHub:</p>
        <p style="text-align: center;"><a href="https://github.com/MMobir/bess-optimization" target="_blank"><img src="https://img.shields.io/badge/GitHub-BESS%20Optimization-blue?style=for-the-badge&logo=github" alt="GitHub Repository"></a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Battery Parameters")
    st.markdown("""
    The system uses environment variables (or reference values from `values.md`) to configure the BESS:
    * **Capacity (MWh)**: Total energy storage capability.
    * **Max Charge/Discharge Power (MW)**: Power rating constraints.
    * **Round-trip Efficiency (%)**: Energy losses during charging/discharging.
    * **Min/Max State of Charge (%)**: Operating range limits.
    * **Initial State of Charge (%)**: Starting energy level.
    * **Battery Cost ($/kWh)** (`BESS_COST_USD_PER_KWH`): Used for degradation cost calculation.
    * **Cycle Life at 100% DoD** (`BESS_CYCLE_LIFE_100PCT_DOD`): Used for degradation cost calculation.
    * *Note: `BESS_CYCLE_LIFE_10PCT_DOD` is no longer used in the current simplified degradation model.*
    """)

# --- App Layout ---
st.title("ERCOT BESS Optimization Dashboard")
st.markdown("Optimize BESS dispatch for Day-Ahead energy arbitrage using Deterministic or Stochastic methods.")

# --- Sidebar Controls ---
st.sidebar.header("Optimization Parameters")

opt_type = st.sidebar.radio(
    "Select Optimization Type",
    ('deterministic', 'stochastic'), 
    captions=["Perfect Foresight", "Considers RTM Uncertainty"])

target_date = st.sidebar.date_input("Select Target Date", value=datetime.now().date() - timedelta(days=1))
settlement_point = st.sidebar.text_input("Settlement Point", value="HB_NORTH")

# Stochastic specific parameters (conditionally displayed)
num_scenarios = 5
noise_std_dev = 5.0
if opt_type == 'stochastic':
    st.sidebar.subheader("Stochastic Parameters")
    num_scenarios = st.sidebar.number_input("Number of RTM Scenarios", min_value=1, max_value=100, value=5, step=1)
    noise_std_dev = st.sidebar.number_input("RTM Price Noise Std Dev ($/MWh)", min_value=0.0, value=40.0, step=5.0)

run_button = st.sidebar.button("Run Optimization")

st.sidebar.header("BESS Configuration")
try:
    bess_config_loaded = load_bess_config()
    st.sidebar.json(bess_config_loaded, expanded=False)
except Exception as e:
    st.sidebar.error(f"Error loading BESS config: {e}")
    bess_config_loaded = None

# --- Main Panel with Tabs --- 
guide_tab, results_tab = st.tabs(["BESS Optimization Guide", "Optimization Results"])

with guide_tab:
    render_bess_guide_tab()
        
with results_tab:
    if run_button:
        if not settlement_point:
            st.error("Please enter a Settlement Point.")
        elif bess_config_loaded is None:
             st.error("Cannot run optimization due to BESS configuration error.")
        else:
            st.markdown(f"### Running {opt_type.capitalize()} Optimization for {target_date} at {settlement_point}")
            if opt_type == 'stochastic':
                st.markdown(f"Using **{num_scenarios}** scenarios with noise std dev **{noise_std_dev} $/MWh**.")
                
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Fetching data and running optimization..."):
                status_text.text("Loading BESS Config...")
                progress_bar.progress(10)
                
                status_text.text("Fetching & Processing DAM Prices...")
                progress_bar.progress(30)
                
                # Call pipeline with selected type and parameters
                results = run_optimization_pipeline(
                    target_date,
                    settlement_point,
                    optimization_type=opt_type,
                    num_scenarios=num_scenarios, 
                    noise_std_dev=noise_std_dev
                )
                progress_bar.progress(80)

            status_text.text("Processing Results...")
            
            if results['success']:
                st.success(f"Optimization Completed Successfully! Status: {results['status']}")
                progress_bar.progress(100)
                status_text.empty()

                # Display results - adapt metric based on type
                if opt_type == 'stochastic':
                    st.metric("Optimal Expected Net Revenue", f"${results['expected_net_revenue']:.2f}")
                    # Add the explanation for stochastic revenue calculation
                    render_revenue_explanation()
                    # Display cost breakdown
                    st.markdown(f"(Expected Gross Revenue: ${results.get('dam_revenue', 0) + results.get('expected_rtm_revenue', 0):.2f}, Expected Degradation Cost: ${results.get('expected_degradation_cost', 0):.2f})")
                else: # Deterministic
                    st.metric("Optimal Net Revenue", f"${results['net_revenue']:.2f}")
                    # Display cost breakdown
                    st.markdown(f"(Gross Revenue: ${results.get('total_revenue', 0):.2f}, Degradation Cost: ${results.get('degradation_cost', 0):.2f})")
                
                # Call the combined chart function (adapts internally)
                st.plotly_chart(create_combined_chart(results['schedule_df'], results['dam_prices_df'], opt_type), use_container_width=True)
                
                st.markdown(f"**Optimal {opt_type.capitalize()} Schedule Data**")
                st.dataframe(format_schedule_display(results['schedule_df'], opt_type), use_container_width=True)
                
                # Always show RTM Scenarios for stochastic optimization (not in an expander)
                if opt_type == 'stochastic' and results.get('rtm_scenario_prices'):
                    st.markdown("### RTM Price Scenarios vs DAM Price")
                    st.markdown("This chart shows how the model considers multiple possible Real-Time Market price scenarios when making Day-Ahead decisions:")
                    scen_fig = go.Figure()
                    # Add DAM price for reference
                    scen_fig.add_trace(go.Scatter(x=results['dam_prices_df']['HourEnding'], y=results['dam_prices_df']['LMP'], name='DAM LMP', line=dict(color='black', width=3)))
                    # Add scenarios
                    for s_idx, rtm_lmp_series in results['rtm_scenario_prices'].items():
                        # Limit number of scenario traces shown in legend for clarity
                        show_legend = s_idx < 5 # Only show first 5 scenarios in legend
                        scen_fig.add_trace(go.Scatter(x=results['dam_prices_df']['HourEnding'], y=rtm_lmp_series, name=f'Scenario {s_idx}', line=dict(dash='dot'), opacity=0.7, showlegend=show_legend))
                    scen_fig.update_layout(
                        title="DAM Price vs. Generated RTM Price Scenarios",
                        xaxis_title="Hour Ending",
                        yaxis_title="LMP ($/MWh)",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h", # Horizontal legend
                            yanchor="bottom",
                            y=-0.25, # Adjusted y-position further down
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=50, r=50, t=80, b=120) # Increased bottom margin
                    )
                    st.plotly_chart(scen_fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="explanation-box">
                        <h4>How to Interpret RTM Scenarios</h4>
                        <p>Each dotted line represents a possible RTM price scenario that could materialize the next day.</p>
                        <p>The optimal DAM schedule (shown in the chart above) positions the battery to take advantage 
                        of expected profitable opportunities across all these scenarios.</p>
                        <p>The scenarios with prices higher than DAM prices present opportunities for discharging 
                        in the RTM, which contributes to the expected revenue calculation.</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"Optimization Failed. Status: {results.get('status', 'N/A')}")
                if results['error_message']:
                    st.error(f"Error Details: {results['error_message']}")
                progress_bar.progress(100)
                status_text.text("Optimization Failed.")

    else:
        st.info("Select optimization parameters, then click 'Run Optimization'.")

# Add GitHub link as a footer at the bottom of the app
st.markdown("""
---
<div style="text-align: center; margin-top: 30px; opacity: 0.7;">
    <p>
        <a href="https://github.com/MMobir/bess-optimization" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-BESS%20Optimization-blue?logo=github" alt="GitHub Repository">
        </a>
    </p>
</div>
""", unsafe_allow_html=True) 