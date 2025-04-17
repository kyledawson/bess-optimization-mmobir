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

def render_revenue_explanation():
    """Renders the explanation of how revenue is calculated in the stochastic model."""
    st.markdown("""
    <div class="explanation-box">
        <h4>Understanding Revenue in Stochastic Optimization</h4>
        <p>The <b>Optimal Expected Revenue</b> shown in the dashboard is calculated using a two-stage stochastic model:</p>
        <ol>
            <li><b>Stage 1 (Day-Ahead Market)</b>: The chart above shows only the Day-Ahead Market (DAM) schedule - when to charge/discharge in the DAM. This is the "here-and-now" decision.</li>
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
    participating in the ERCOT electricity market. The core objective is to **maximize revenue** by determining the 
    optimal times to charge (buy) and discharge (sell) energy, considering both Day-Ahead Market (DAM) and 
    Real-Time Market (RTM) price variations.
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
        <p>Formulates a linear programming problem using PuLP to maximize expected revenue:</p>
        <ul>
            <li><b>Decision Variables</b>: When and how much to charge/discharge in the DAM, and what adjustments to make in each RTM scenario</li>
            <li><b>Constraints</b>: Battery capacity, power limits, state of charge limits, and energy conservation equations</li>
            <li><b>Objective</b>: Maximize expected profit across all scenarios (weighted by probability)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-box">
        <h4>4. Result Interpretation</h4>
        <p>The "Optimal Expected Revenue" represents the weighted average profit across all RTM scenarios, 
        given the optimal DAM bidding strategy. Even if the DAM schedule shows only charging, the expected 
        revenue includes anticipated profits from discharging in the RTM under various price scenarios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Technical Architecture")
    
    st.markdown("""
    * **bess_optimizer.py**: Core optimization logic, scenario generation, and API integration
    * **ercot_api_fetcher.py**: Functions to fetch and process ERCOT market data
    * **streamlit_app.py**: Interactive web dashboard for visualization and parameter settings
    """)
    
    st.subheader("Battery Parameters")
    st.markdown("""
    The system uses environment variables to configure the BESS parameters:
    * Capacity (MWh): Total energy storage capability
    * Maximum charge/discharge power (MW): Power rating constraints
    * Round-trip efficiency (%): Energy losses during charging/discharging cycle
    * Minimum/maximum state of charge (%): Operating range limits to preserve battery life
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
                    st.metric("Optimal Expected Revenue", f"${results['expected_revenue']:.2f}")
                    # Add the explanation for stochastic revenue calculation
                    render_revenue_explanation()
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