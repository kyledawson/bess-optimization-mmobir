import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the refactored optimizer functions
# Note: Ensure bess_optimizer.py has been updated for stochastic mode
from bess_optimizer import run_optimization_pipeline, load_bess_config

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="BESS Optimization Dashboard")

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
    soc_col = 'SoC_Percent_End' # Only available for deterministic in current setup

    # Check if essential columns exist in schedule_df
    required_sched_cols = [charge_col, discharge_col]
    if not is_stochastic:
        required_sched_cols.append(soc_col)
    if not all(col in schedule_df.columns for col in required_sched_cols):
        st.warning(f"Schedule DataFrame missing required columns for charting: {required_sched_cols}")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.update_layout(title_text="Price and Dispatch Schedule (Schedule Data Incomplete)", height=600)
        return fig
        
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.3, 0.7], 
        specs=[[{"secondary_y": False}], # Row 1 (Price)
               [{"secondary_y": not is_stochastic}]] # Row 2 (Schedule), add secondary axis only if SoC is plotted
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
    # State of Charge (%) - Only for Deterministic currently
    if not is_stochastic:
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
        legend_title_text="Metrics",
        hovermode="x unified"
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="LMP ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1, secondary_y=False)
    if not is_stochastic:
         fig.update_yaxes(title_text="SoC (%)", row=2, col=1, secondary_y=True, range=[0, 100], showgrid=False)
    fig.update_xaxes(title_text="Hour Ending", row=2, col=1)

    return fig

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
    noise_std_dev = st.sidebar.number_input("RTM Price Noise Std Dev ($/MWh)", min_value=0.0, value=10.0, step=0.5)

run_button = st.sidebar.button("Run Optimization")

st.sidebar.header("BESS Configuration")
try:
    bess_config_loaded = load_bess_config()
    st.sidebar.json(bess_config_loaded, expanded=False)
except Exception as e:
    st.sidebar.error(f"Error loading BESS config: {e}")
    bess_config_loaded = None

# --- Main Panel --- 
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
                st.metric("Optimal Expected Revenue", f"${results['expected_revenue']:.2f}")
                st.markdown("_Note: This is the DAM schedule optimized considering RTM price uncertainty._")
            else:
                st.metric("Optimal Deterministic Revenue", f"${results['total_revenue']:.2f}")
            
            # Call the combined chart function (adapts internally)
            st.plotly_chart(create_combined_chart(results['schedule_df'], results['dam_prices_df'], opt_type), use_container_width=True)
            
            st.markdown(f"**Optimal {opt_type.capitalize()} Schedule Data**")
            st.dataframe(format_schedule_display(results['schedule_df'], opt_type), use_container_width=True)
            
            # Optionally display RTM Scenarios if stochastic
            if opt_type == 'stochastic' and results.get('rtm_scenario_prices'):
                 with st.expander("View Generated RTM Price Scenarios"):
                     scen_fig = go.Figure()
                     # Add DAM price for reference
                     scen_fig.add_trace(go.Scatter(x=results['dam_prices_df']['HourEnding'], y=results['dam_prices_df']['LMP'], name='DAM LMP', line=dict(color='black', width=3)))
                     # Add scenarios
                     for s_idx, rtm_lmp_series in results['rtm_scenario_prices'].items():
                         scen_fig.add_trace(go.Scatter(x=results['dam_prices_df']['HourEnding'], y=rtm_lmp_series, name=f'Scenario {s_idx}', line=dict(dash='dot'), opacity=0.7))
                     scen_fig.update_layout(title="DAM Price vs. Generated RTM Price Scenarios", xaxis_title="Hour Ending", yaxis_title="LMP ($/MWh)")
                     st.plotly_chart(scen_fig, use_container_width=True)

        else:
            st.error(f"Optimization Failed. Status: {results.get('status', 'N/A')}")
            if results['error_message']:
                st.error(f"Error Details: {results['error_message']}")
            progress_bar.progress(100)
            status_text.text("Optimization Failed.")

else:
    st.info("Select optimization parameters, then click 'Run Optimization'.") 