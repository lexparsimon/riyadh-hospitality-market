import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Optional: Cache the simulation so that if parameters haven't changed, it reuses the result.
@st.cache_data
def simulate_market(demand_growth, inflation_rate, supply5_str, supply4_str):
    T = 11  # Simulate from 2024 to 2034.
    years = np.arange(2024, 2024 + T)
    
    # Convert inputs and set parameters
    demand_growth = float(demand_growth)
    inflation_rate = float(inflation_rate)
    inflation = inflation_rate
    high_demand = (demand_growth > 0.10)
    
    # Initial conditions
    K1_0 = 7550.0; K2_0 = 6124.0; K3_0 = 3266.0
    ADR1_0 = 1324.0; ADR2_0 = 1137.0; ADR3_0 = 437.0
    target_occ = {'group1': 0.65, 'group2': 0.40, 'group3': 0.65}
    D0 = (K1_0 * target_occ['group1'] + K2_0 * target_occ['group2'] + K3_0 * target_occ['group3']) * 365
    
    try:
        supply5_list = [float(x.strip()) for x in supply5_str.split(',')]
        supply4_list = [float(x.strip()) for x in supply4_str.split(',')]
    except Exception as e:
        st.error("Error parsing supply strings. Please enter comma-separated numbers.")
        return None

    new_supply_5 = np.array(supply5_list)
    new_supply_4 = np.array(supply4_list)
    if len(new_supply_5) < T:
        new_supply_5 = np.concatenate([new_supply_5, np.zeros(T - len(new_supply_5))])
    else:
        new_supply_5 = new_supply_5[:T]
    if len(new_supply_4) < T:
        new_supply_4 = np.concatenate([new_supply_4, np.zeros(T - len(new_supply_4))])
    else:
        new_supply_4 = new_supply_4[:T]

    # Initialize state variables
    K1 = np.zeros(T); K2 = np.zeros(T); K3 = np.zeros(T)
    K1[0] = K1_0; K2[0] = K2_0; K3[0] = K3_0
    Potential1 = np.zeros(T); Potential1[0] = K1_0
    for t in range(1, T):
        Potential1[t] = Potential1[t - 1] + new_supply_5[t]

    ADR1_real = np.zeros(T); ADR2_real = np.zeros(T); ADR3_real = np.zeros(T)
    ADR1_real[0] = ADR1_0; ADR2_real[0] = ADR2_0; ADR3_real[0] = ADR3_0
    ADR1_nom = ADR1_real.copy(); ADR2_nom = ADR2_real.copy(); ADR3_nom = ADR3_real.copy()

    # Demand allocation multipliers
    M1 = 1.2; M2 = 0.8; M3 = 1.0
    k1 = 0.30; k2 = 0.40
    def migration_flow(occ, target_occ_val, keys_current, k):
        shortfall = max(target_occ_val - occ, 0) / target_occ_val
        return k * shortfall * keys_current

    alpha = 0.05; delta2 = 0.06; beta = 1.0; beta2 = 0.9

    for t in range(1, T):
        K1[t] = K1[t - 1] + new_supply_5[t]
        K2[t] = K2[t - 1] + new_supply_4[t]
        K3[t] = K3[t - 1]
        D_total = D0 * ((1 + demand_growth) ** t)
        W1 = M1 * (K1[t] / ADR1_nom[t - 1])
        W2 = M2 * (K2[t] / ADR2_nom[t - 1])
        W3 = M3 * (K3[t] / ADR3_nom[t - 1])
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
            m1 = 0.0; m2 = 0.0
        K1[t] = K1[t] - m1
        K2[t] = K2[t] + m1 - m2
        K3[t] = K3[t] + m2
        if not high_demand:
            scarcity = 1 - (K1[t] / Potential1[t])
            ADR1_real[t] = ADR1_real[t - 1] * (1 + inflation + alpha * scarcity)
        else:
            ADR1_real[t] = ADR1_real[t - 1] * (1 + inflation + beta * (demand_growth - 0.10))
        if not high_demand:
            pressure = max(target_occ['group2'] - occ2, 0) / target_occ['group2']
            ADR2_real[t] = ADR2_real[t - 1] * (1 + inflation) * (1 - delta2 * pressure)
        else:
            ADR2_real[t] = ADR2_real[t - 1] * (1 + inflation + beta2 * (demand_growth - 0.10))
        ADR3_real[t] = ADR3_real[t - 1] * (1 + inflation)
        ADR1_nom[t] = ADR1_real[t]
        ADR2_nom[t] = ADR2_real[t]
        ADR3_nom[t] = ADR3_real[t]

    total_keys = K1 + K2 + K3
    share1 = 100 * K1 / total_keys
    share2 = 100 * K2 / total_keys
    share3 = 100 * K3 / total_keys

    # Create the charts
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(years, share1, marker='o', color='#1f77b4', linewidth=2, label='Group 1 (Luxury)')
    ax[0].plot(years, share2, marker='o', color='#ff7f0e', linewidth=2, label='Group 2 (Upper-Mid)')
    ax[0].plot(years, share3, marker='o', color='#2ca02c', linewidth=2, label='Group 3 (Budget)')
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Market Share (%)")
    ax[0].set_title("Evolution of Market Shares")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(years, ADR1_nom, marker='o', color='#d62728', linewidth=2, label='Group 1 ADR')
    ax[1].plot(years, ADR2_nom, marker='o', color='#9467bd', linewidth=2, label='Group 2 ADR')
    ax[1].plot(years, ADR3_nom, marker='o', color='#8c564b', linewidth=2, label='Group 3 ADR')
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("ADR (SAR, nominal)")
    ax[1].set_title("Evolution of Nominal ADR")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    return fig

# Build the Streamlit interface
st.title("Riyadh Hospitality Market")
st.subheader("2024-2034 Supply-Demand Dynamics")

# Input widgets: changes in any widget will automatically trigger a rerun.
demand_growth = st.slider("Demand Growth (%)", 0.0, 25.0, 5.0) / 100.0
inflation_rate = st.slider("Inflation (%)", 1.0, 20.0, 2.0) / 100.0
supply5_str = st.text_input("Supply 5", "860,1168,2408,4586,1945,481,490,0,0,0,384")
supply4_str = st.text_input("Supply 4", "417,1317,1281,1170,950,384,224,0,0,294,0")

# Automatically run the simulation when any input changes.
fig = simulate_market(demand_growth, inflation_rate, supply5_str, supply4_str)
if fig is not None:
    st.pyplot(fig)
