import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components  # For embedding external HTML

# Set page configuration to wide.
st.set_page_config(page_title="Riyadh Hospitality Market", layout="wide")

# -----------------------------------------------------------------------------
# Inject custom CSS for mobile responsiveness
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Reduce padding and use full width on mobile */
    .reportview-container .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    /* Force columns to stack on small screens */
    @media (max-width: 768px) {
        div[data-testid="column"] {
            flex: 100% !important;
            max-width: 100% !important;
            display: block;
            margin-bottom: 1rem;
        }
    }
    /* Make the embedded HTML (e.g., Kepler map) responsive */
    .responsive-html {
        width: 100%;
        overflow-x: auto;
    }
    /* Adjust header sizes for mobile */
    @media (max-width: 768px) {
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1. Title and Subtitle
# =============================================================================
st.title("Riyadh Hospitality Market")
st.subheader("2024-2034 Supply-Demand Dynamics")

# =============================================================================
# 2. Plotly Chart: Hotel Quadrants
# =============================================================================
@st.cache_data
def load_hotel_data():
    # Ensure that "Riyadh_hotel_data.csv" is in your repository.
    return pd.read_csv("Riyadh_hotel_data.csv")

df_filtered = load_hotel_data()
adr_median = df_filtered['median_adr_sar'].median()
occupancy_median = df_filtered['occupancy_rate'].median()

colors = {
    "High ADR & High Occupancy": "green",
    "High ADR & Low Occupancy": "blue",
    "Low ADR & High Occupancy": "orange",
    "Low ADR & Low Occupancy": "red"
}
df_filtered['color'] = df_filtered['quadrant'].map(colors)

fig_plotly = px.scatter(
    df_filtered,
    x='median_adr_sar',
    y='occupancy_rate',
    color='quadrant',
    size='number_of_keys',
    hover_name='name',
    custom_data=["number_of_keys", "parent_chain"],
    title="Hotel Quadrants: ADR vs Occupancy (Size by Number of Keys)",
    labels={
        "median_adr_sar": "Median ADR (SAR)",
        "occupancy_rate": "Occupancy Rate",
        "number_of_keys": "Number of Keys",
        "parent_chain": "Parent Chain"
    }
)

fig_plotly.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br><br>" +
        "Median ADR (SAR) = %{x}<br>" +
        "Occupancy Rate = %{y}<br>" +
        "Number of Keys = %{customdata[0]}<br>" +
        "Parent Chain = %{customdata[1]}<extra></extra>"
    )
)

fig_plotly.add_vline(
    x=adr_median,
    line_dash="dash",
    line_color="rgba(255,255,255,0.3)",
    line_width=1,
    annotation_text="Median ADR"
)
fig_plotly.add_hline(
    y=occupancy_median,
    line_dash="dash",
    line_color="rgba(255,255,255,0.3)",
    line_width=1,
    annotation_text="Median Occupancy"
)

fig_plotly.update_layout(
    xaxis_title="Median ADR (SAR)",
    yaxis_title="Occupancy Rate",
    legend_title="Quadrant",
    template="plotly_dark"
)

st.plotly_chart(fig_plotly, use_container_width=True)

# =============================================================================
# 3. Kepler.gl Map Section
# =============================================================================
st.subheader("Map of Hotels")
try:
    with open("riyadh_hospitality.html", "r", encoding="utf-8") as f:
        kepler_map_html = f.read()
    # Wrap the embedded HTML in a responsive div.
    st.markdown('<div class="responsive-html">', unsafe_allow_html=True)
    components.html(kepler_map_html, height=600, width=1000)
    st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.error("Error loading Kepler map. Please ensure 'kepler_map.html' is in the repository.")

# =============================================================================
# 4. Occupancy Projection Graph for Group 1 (5‑star Luxury)
# =============================================================================
@st.cache_data
def plot_occupancy_projection():
    # Starting values
    keys_today = 7550
    occ_rate_today = 0.7121
    occupied_today = keys_today * occ_rate_today  # ≈5375

    # Upcoming 5‑star supply numbers for Group 1 for 2025–2035
    new_supply_group1 = [860, 1168, 2408, 4586, 1945, 481, 490, 0, 0, 0, 384]

    # Compute cumulative supply (starting with keys_today at 2025)
    cumulative_supply_group1 = [keys_today]
    for ns in new_supply_group1:
        cumulative_supply_group1.append(cumulative_supply_group1[-1] + ns)
    supply = np.array(cumulative_supply_group1[:-1])  # For years 2025 to 2035

    years_proj = np.arange(0, len(supply))  # 0,...,10
    demand_growth_rates = [0.00, 0.05, 0.10, 0.15, 0.20]

    # Create a figure with a dark background matching Streamlit's (#0E1117)
    fig_occ, ax_occ = plt.subplots(figsize=(20,10))
    fig_occ.patch.set_facecolor('#0E1117')
    ax_occ.set_facecolor('#0E1117')

    # Use dimmed white for all text (70% opacity white)
    text_color = (1, 1, 1, 0.7)

    for g in demand_growth_rates:
        demand = occupied_today * (1 + g) ** years_proj
        occupancy = 100 * demand / supply
        ax_occ.plot(2025 + years_proj, occupancy, marker='o', linewidth=2, label=f"Demand Growth = {g*100:.0f}%/yr")
    ax_occ.axhline(occ_rate_today*100, color='red', linestyle='--', linewidth=1, label=f"Current Occupancy ({occ_rate_today*100:.1f}%)")
    ax_occ.set_xlabel("Year", fontsize=18, color=text_color)
    ax_occ.set_ylabel("Occupancy Rate (%)", fontsize=18, color=text_color)
    ax_occ.set_title("Projected Occupancy, High ADR & High Occupancy (Premium), 2025–2034", fontsize=20, color=text_color)
    ax_occ.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax_occ.set_ylim(0, 100.1)
    ax_occ.tick_params(axis='x', labelsize=16, colors=text_color)
    ax_occ.tick_params(axis='y', labelsize=16, colors=text_color)
    for spine in ax_occ.spines.values():
        spine.set_visible(False)
    leg_occ = ax_occ.legend(fontsize=14, frameon=False)
    for txt in leg_occ.get_texts():
        txt.set_color(text_color)
    plt.tight_layout()
    return fig_occ

st.subheader("Projected Occupancy for High ADR & High Occupancy (Premium) Segment, 2025-2034")
fig_occ = plot_occupancy_projection()
st.pyplot(fig_occ)

# =============================================================================
# 5. Simulation Graph and Controls
# =============================================================================
@st.cache_data
def simulate_market(demand_growth, inflation_rate, upcoming_supply_total):
    T = 11  # Simulation for years 2024 to 2034.
    years = np.arange(2024, 2024 + T)
    
    default5 = np.array([1168, 2408, 4586, 1945, 481, 490, 0, 0, 0, 384])
    default4 = np.array([1317, 1281, 1170, 950, 384, 224, 0, 0, 294, 0])
    total_default = np.sum(default5 + default4)
    
    new_supply_5 = np.zeros(T)
    new_supply_4 = np.zeros(T)
    for t in range(1, T):
        new_supply_5[t] = (default5[t-1] / total_default) * upcoming_supply_total
        new_supply_4[t] = (default4[t-1] / total_default) * upcoming_supply_total

    demand_growth = float(demand_growth)
    inflation_rate = float(inflation_rate)
    inflation = inflation_rate
    
    # Compute a dynamic threshold for high demand.
    # Baseline threshold is 10% when upcoming_supply_total equals 17082.
    baseline_supply = 17082
    threshold = 0.10 * (upcoming_supply_total / baseline_supply)
    high_demand = (demand_growth > threshold)
    
    K1_0, K2_0, K3_0 = 7550.0, 6124.0, 3266.0
    ADR1_0, ADR2_0, ADR3_0 = 1324.0, 1137.0, 437.0
    target_occ = {'group1': 0.65, 'group2': 0.40, 'group3': 0.65}
    D0 = (K1_0 * target_occ['group1'] + K2_0 * target_occ['group2'] + K3_0 * target_occ['group3']) * 365
    
    K1 = np.zeros(T); K2 = np.zeros(T); K3 = np.zeros(T)
    K1[0], K2[0], K3[0] = K1_0, K2_0, K3_0
    
    Potential1 = np.zeros(T)
    Potential1[0] = K1_0
    for t in range(1, T):
        Potential1[t] = Potential1[t-1] + new_supply_5[t]
    
    ADR1_real = np.zeros(T); ADR2_real = np.zeros(T); ADR3_real = np.zeros(T)
    ADR1_real[0], ADR2_real[0], ADR3_real[0] = ADR1_0, ADR2_0, ADR3_0
    ADR1_nom = ADR1_real.copy(); ADR2_nom = ADR2_real.copy(); ADR3_nom = ADR3_real.copy()
    
    M1, M2, M3 = 1.2, 0.8, 1.0
    k1, k2 = 0.30, 0.40
    def migration_flow(occ, target_occ_val, keys_current, k):
        shortfall = max(target_occ_val - occ, 0) / target_occ_val
        return k * shortfall * keys_current
    
    alpha, delta2, beta, beta2 = 0.05, 0.06, 1.0, 0.9
    for t in range(1, T):
        K1[t] = K1[t-1] + new_supply_5[t]
        K2[t] = K2[t-1] + new_supply_4[t]
        K3[t] = K3[t-1]
        D_total = D0 * ((1 + demand_growth) ** t)
        W1 = M1 * (K1[t] / ADR1_nom[t-1])
        W2 = M2 * (K2[t] / ADR2_nom[t-1])
        W3 = M3 * (K3[t] / ADR3_nom[t-1])
        W_sum = W1 + W2 + W3
        D1 = D_total * (W1 / W_sum)
        D2 = D_total * (W2 / W_sum)
        D3 = D_total * (W3 / W_sum)
        occ1 = min(D1 / (K1[t] * 365), 1.0)
        occ2 = min(D2 / (K2[t] * 365), 1.0)
        occ3 = min(D3 / (K3[t] * 365), 1.0)
        if not high_demand:
            m1 = migration_flow(occ1, target_occ['group1'], K1[t], k1)
            m2 = migration_flow(occ2, target_occ['group2'], K2[t], k2)
        else:
            m1, m2 = 0.0, 0.0
        K1[t] = K1[t] - m1
        K2[t] = K2[t] + m1 - m2
        K3[t] = K3[t] + m2
        if not high_demand:
            scarcity = 1 - (K1[t] / Potential1[t])
            ADR1_real[t] = ADR1_real[t-1] * (1 + inflation + alpha * scarcity)
        else:
            ADR1_real[t] = ADR1_real[t-1] * (1 + inflation + beta * (demand_growth - 0.10))
        if not high_demand:
            pressure = max(target_occ['group2'] - occ2, 0) / target_occ['group2']
            ADR2_real[t] = ADR2_real[t-1] * (1 + inflation) * (1 - delta2 * pressure)
        else:
            ADR2_real[t] = ADR2_real[t-1] * (1 + inflation + beta2 * (demand_growth - 0.10))
        ADR3_real[t] = ADR3_real[t-1] * (1 + inflation)
        ADR1_nom[t] = ADR1_real[t]
        ADR2_nom[t] = ADR2_real[t]
        ADR3_nom[t] = ADR3_real[t]
    
    total_keys = K1 + K2 + K3
    share1 = 100 * K1 / total_keys
    share2 = 100 * K2 / total_keys
    share3 = 100 * K3 / total_keys
    
    try:
        plt.style.use('dark_background')
    except OSError:
        plt.style.use('default')
    
    fig_sim, ax_sim = plt.subplots(1, 2, figsize=(20, 10))
    fig_sim.patch.set_facecolor('#0E1117')
    
    text_color = (1, 1, 1, 0.7)  # Dimmed white
    
    ax_sim[0].plot(years + 2024, share1, marker='o', color='#1f77b4', linewidth=2,
                   label='High ADR & High Occupancy (Premium)')
    ax_sim[0].plot(years + 2024, share2, marker='o', color='#ff7f0e', linewidth=2,
                   label='High ADR & Low Occupancy (Upper-Mid)')
    ax_sim[0].plot(years + 2024, share3, marker='o', color='#2ca02c', linewidth=2,
                   label='Low ADR & High Occupancy (Budget)')
    ax_sim[0].set_xlabel("Year", fontsize=18, color=text_color)
    ax_sim[0].set_ylabel("Market Share (%)", fontsize=18, color=text_color)
    ax_sim[0].set_title("Evolution of Market Shares", fontsize=20, color=text_color)
    leg_sim1 = ax_sim[0].legend(fontsize=14, frameon=False)
    for t in leg_sim1.get_texts():
        t.set_color(text_color)
    ax_sim[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax_sim[0].set_facecolor('#0E1117')
    ax_sim[0].tick_params(axis='x', labelsize=16, colors=text_color)
    ax_sim[0].tick_params(axis='y', labelsize=16, colors=text_color)
    
    ax_sim[1].plot(years + 2024, ADR1_nom, marker='o', color='#d62728', linewidth=2,
                   label='High ADR & High Occupancy (Premium)')
    ax_sim[1].plot(years + 2024, ADR2_nom, marker='o', color='#9467bd', linewidth=2,
                   label='High ADR & Low Occupancy (Upper-Mid)')
    ax_sim[1].plot(years + 2024, ADR3_nom, marker='o', color='#8c564b', linewidth=2,
                   label='Low ADR & High Occupancy (Budget)')
    ax_sim[1].set_xlabel("Year", fontsize=18, color=text_color)
    ax_sim[1].set_ylabel("ADR (SAR, nominal)", fontsize=18, color=text_color)
    ax_sim[1].set_title("Evolution of Nominal ADR", fontsize=20, color=text_color)
    leg_sim2 = ax_sim[1].legend(fontsize=14, frameon=False)
    for t in leg_sim2.get_texts():
        t.set_color(text_color)
    ax_sim[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax_sim[1].set_facecolor('#0E1117')
    ax_sim[1].tick_params(axis='x', labelsize=16, colors=text_color)
    ax_sim[1].tick_params(axis='y', labelsize=16, colors=text_color)
    
    for a in ax_sim:
        for spine in a.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    return fig_sim

# st.subheader("Simulation Output")
# Make the simulation output column wider by adjusting the columns ratio (e.g., 5:1).
col1, col2 = st.columns([5, 1])
with col2:
    st.markdown("## Controls")
    demand_growth = st.slider("Demand Growth Rate (%)", 0.0, 25.0, 5.0, step=0.1) / 100.0
    inflation_rate = st.slider("Inflation (%)", 1.0, 20.0, 2.0, step=0.1) / 100.0
    upcoming_supply_total = st.slider("Upcoming Supply 2025-2034", 0, 50000, 17082, step=100)
with col1:
    st.markdown("## Simulation Output")
    fig_sim = simulate_market(demand_growth, inflation_rate, upcoming_supply_total)
    if fig_sim is not None:
        st.pyplot(fig_sim)
