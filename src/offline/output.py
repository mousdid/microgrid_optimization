import pandas as pd
import matplotlib.pyplot as plt

def save_report(report: dict, filename: str = 'output/report.csv'):
    df = pd.DataFrame(report)
    df.index.name = 'Time'
    df.to_csv(filename)
    print(f"Report saved to {filename}")

def plot_results(report: dict):
    # Check if input is already a DataFrame (from CSV) or a dict (from solver)
    if isinstance(report, pd.DataFrame):
        df = report
    else:
        df = pd.DataFrame(report)
    
    # Figure 1: Electric balance plot with battery SOC
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Plot generation sources on primary axis
    print("Columns in df:", df.columns.tolist())

    df[['CHP','DG','PV','WT']].plot.area(ax=ax1, alpha=0.6)
    ax1.plot(df['Load_el'], 'k--', label='Electric Load')
    ax1.plot(df['Import'], 'r-', label='Grid Import')
    ax1.set_title("Electric Balance and Battery SOC")
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlabel("Hour")
    ax1.grid(True)
    
    # Create secondary y-axis for battery SOC
    ax2 = ax1.twinx()
    
    # Calculate SOC percentage (assuming max battery capacity is max value in SOC column)
    max_capacity = df['SOC'].max()
    soc_percentage = df['SOC'] / max_capacity * 100
    
    # Plot SOC percentage on secondary axis
    ax2.plot(soc_percentage, 'g-', linewidth=2, label='Battery SOC %')
    ax2.set_ylabel("Battery SOC (%)")
    ax2.set_ylim(0, 105)  # Set y-axis limits for percentage
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/electric_balance_with_soc.png')
    plt.show()  # Show Figure 1 and wait for close
   
    # Figure 2: Thermal balance plot
    plt.figure(figsize=(10, 3))
    plt.plot(df['Heat_CHP'], label='CHP Heat')
    plt.plot(df['Load_th'], '--', label='Heat Demand')
    plt.title("Thermal Balance")
    plt.ylabel("kW_th")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True)
    plt.savefig('output/thermal_balance.png')
    plt.show()  # Show Figure 2 and wait for close

    # Figure 3: State of charge plots
    plt.figure(figsize=(10, 3))
    plt.plot(df['SOC'], label='Battery SOC')
    plt.plot(df['EV_SOC'], label='EV SOC')
    plt.title("State of Charge")
    plt.ylabel("kWh")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True)
    plt.savefig('output/state_of_charge.png')
    plt.show()  # Show Figure 3 and wait for close

    # Figure 4: Binary state plots
    plt.figure(figsize=(10, 4))
    binary_cols = [col for col in ['u_CHP', 'u_DG', 'u_CH_ES', 'u_DIS_ES', 'A_EV'] if col in df.columns]
    if binary_cols:
        df[binary_cols].astype(float).plot(drawstyle='steps-post')
        plt.title("Unit Status States")
        plt.ylabel("Binary State")
        plt.xlabel("Hour")
        plt.yticks([0, 1], ['Off', 'On'])
        plt.legend()
        plt.grid(True)
        plt.savefig('output/binary_states.png')
        plt.show()  # Show Figure 4 and wait for close
    else:
        print("No binary state data available for plotting")

    # Figure 5: Startup indicator for DG
    plt.figure(figsize=(10, 2))
    df['startup_DG'].astype(float).plot(kind='bar', color='orange')
    plt.title("DG Startup Indicator")
    plt.ylabel("Startup Event")
    plt.xlabel("Hour")
    plt.grid(True)
    plt.savefig('output/dg_startup.png')
    plt.show()  # Show Figure 5 and wait for close

