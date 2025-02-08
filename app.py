import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def simulate_market(demand_growth, inflation_rate, upcoming_supply_total):
    T = 11  # Simulation for years 2024 to 2034.
    years = np.arange(2024, 2024 + T)
    
    # --- Upcoming Supply Distribution ---
    # Default upcoming supply values (for years 2025-2034) from your original data:
    default5 = np.array([1168, 2408, 4586, 1945, 481, 490, 0, 0, 0, 384])
    default4 = np.array([1317, 1281, 1170, 950, 384, 224, 0, 0, 294, 0])
    total_default = np.sum(default5 + default4)  # Total default upcoming supply for 2025-2034
    
    # Create upcoming supply arrays for each year (year 2024 gets no new supply).
    new_supply_5 = np.zeros(T)
    new_supply_4 = np.zeros(T)
    for t in range(1, T):
        new_supply_5[t] = (default5[t-1] / total_default) * upcoming_supply_total
        new_supply_4[t] = (default4[t-1] / total_default) * upcoming_supply_total

    # --- Simulation Settings ---
    demand_growth = float(demand_growth)
    inflation_rate = float(inflation_rate)
    inflation = inflation_rate
    high_demand = (demand_growth > 0.10)
    
    # Initial conditions
    K1_0 = 7550.0  # Group 1 keys (Luxury)
    K2_0 = 6124.0  # Group 2 keys (Upper-Mid)
    K3_0 = 3266.0  # Group 3 keys (Budget)
    
    ADR1_0 = 1324.0  # Initial ADR for Group 1
    ADR2_0 = 1137.0  # Initial ADR for Group 2
    ADR3_0 = 437.0   # Initial ADR for Group 3
    
    target_occ = {'group1': 0.65, 'group2': 0.40, 'group3': 0.65}
    D0 = (K1_0 * target_occ['group1'] + K2_0 * target_occ['group2'] + K3_0 * target_occ['group3']) * 365
    
    # Initialize state arrays for keys.
    K1 = np.zeros(T); K2 = np.zeros(T); K3 = np.zeros(T)
    K1[0] = K1_0; K2[0] = K2_0; K3[0] = K3_0
    
    Potential1 = np.zeros(T)
    Potential1[0] = K1_0
    for t in range(1, T):
        Potential1[t] = Potential1[t-1] + new_supply_5[t]
    
    ADR1_real = np.zeros(T); ADR2_real = np.zeros(T); ADR3_real = np.zeros(T)
    ADR1_real[0] = ADR1_0; ADR2_real[0] = ADR2_0; ADR3_real[0] = ADR3_0
    ADR1_nom = ADR1_real.copy(); ADR2_nom = ADR2_real.copy(); ADR3_nom = ADR3_real.copy()
    
    # Demand allocation multipliers and migration parameters.
    M1 = 1.2; M2 = 0.8; M3 = 1.0
    k1 = 0.30; k2 = 0.40
    def migration_flow(occ, target_occ_val, keys_current, k):
        shortfall = max(target_occ_val - occ, 0) / target_occ_val
        return k * shortfall * keys_current
    
    alpha = 0.05; delta2 = 0.06; beta = 1.0; beta2 = 0.9
    
    for t in range(1, T):
        # (A) Add new supply.
        K1[t] = K1[t-1] + new_supply_5[t]
        K2[t] = K2[t-1] + new_supply_4[t]
        K3[t] = K3[t-1]
        
        # (B) Compute total market demand for period t.
        D_total = D0 * ((1 + demand_growth) ** t)
        
        # (C) Allocate demand using attractiveness weights.
        W1 = M1 * (K1[t] / ADR1_nom[t-1])
        W2 = M2 * (K2[t] / ADR2_nom[t-1])
        W3 = M3 * (K3[t] / ADR3_nom[t-1])
        W_sum = W1 + W2 + W3
        
        D1 = D_total * (W1 / W_sum)
        D2 = D_total * (W2 / W_sum)
        D3 = D_total * (W3 / W_sum)
        
        # (D) Compute occupancy.
        occ1 = min(D1 / (K1[t] * 365), 1.0)
        occ2 = min(D2 / (K2[t] * 365), 1.0)
        occ3 = min(D3 / (K3[t] * 365), 1.0)
        
        # (E) Migration flows (if not high demand).
        if not high_demand:
            m1 = migration_flow(occ1, target_occ['group1'], K1[t], k1)
            m2 = migration_flow(occ2, target_occ['group2'], K2[t], k2)
        else:
            m1 = 0.0; m2 = 0.0
        
        K1[t] = K1[t] - m1
        K2[t] = K2[t] + m1 - m2
        K3[t] = K3[t] + m2
        
        # (F) Update ADRs.
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
    
    # --- Set a dark style.
    try:
        plt.style.use('dark_background')
    except OSError:
        plt.style.use('default')
    
    # Create larger graphs with the dark style and no frame.
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Market Shares
    ax[0].plot(years, share1, marker='o', color='#1f77b4', linewidth=2, label='Group 1 (Luxury)')
    ax[0].plot(years, share2, marker='o', color='#ff7f0e', linewidth=2, label='Group 2 (Upper-Mid)')
    ax[0].plot(years, share3, marker='o', color='#2ca02c', linewidth=2, label='Group 3 (Budget)')
    ax[0].set_xlabel("Year", fontsize=16)
    ax[0].set_ylabel("Market Share (%)", fontsize=16)
    ax[0].set_title("Evolution of Market Shares", fontsize=18)
    ax[0].legend(fontsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot ADRs
    ax[1].plot(years, ADR1_nom, marker='o', color='#d62728', linewidth=2, label='Group 1 ADR')
    ax[1].plot(years, ADR2_nom, marker='o', color='#9467bd', linewidth=2, label='Group 2 ADR')
    ax[1].plot(years, ADR3_nom, marker='o', color='#8c564b', linewidth=2, label='Group 3 ADR')
    ax[1].set_xlabel("Year", fontsize=16)
    ax[1].set_ylabel("ADR (SAR, nominal)", fontsize=16)
    ax[1].set_title("Evolution of Nominal ADR", fontsize=18)
    ax[1].legend(fontsize=14)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    
    # --- Remove frames (axes spines) for a cool, frameless look.
    for a in ax:
        for spine in a.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    return fig

# --- Layout Setup ---
st.title("Riyadh Hospitality Market")
st.subheader("2024-2034 Supply-Demand Dynamics")

# Create two columns: a wide left column for the graph and a narrow right column for controls.
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("## Controls")
    # Slider for Demand Growth Rate
    demand_growth = st.slider("Demand Growth Rate (%)", 0.0, 25.0, 5.0, step=0.1) / 100.0
    # Slider for Inflation
    inflation_rate = st.slider("Inflation (%)", 1.0, 20.0, 2.0, step=0.1) / 100.0
    # Slider for total upcoming supply (for 2025-2034)
    upcoming_supply_total = st.slider("Upcoming Supply 2025-2034", 0, 50000, 17082, step=100)
    
with col1:
    st.markdown("## Simulation Output")
    fig = simulate_market(demand_growth, inflation_rate, upcoming_supply_total)
    if fig is not None:
        st.pyplot(fig)
