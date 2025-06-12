import pandas as pd
import matplotlib.pyplot as plt

def save_report(report: dict, filename: str = 'output/report.csv'):
    df = pd.DataFrame(report).T
    df.index.name = 'Time'
    df.to_csv(filename)
    print(f"Report saved to {filename}")

def plot_results(report: dict):
    df = pd.DataFrame(report).T

    # Electric balance plot
    plt.figure(figsize=(10, 4))
    df[['CHP','DG','PV','WT']].plot.area(ax=plt.gca(), alpha=0.6)
    plt.plot(df['Load_el'], 'k--', label='Electric Load')
    plt.plot(df['Import'], 'r-', label='Grid Import')
    plt.title("Electric Balance")
    plt.ylabel("kW")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True)

    # Thermal balance plot
    plt.figure(figsize=(10, 3))
    plt.plot(df['Heat_CHP'], label='CHP Heat')
    plt.plot(df['Load_th'], '--', label='Heat Demand')
    plt.title("Thermal Balance")
    plt.ylabel("kW_th")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True)

    # State of charge plots
    plt.figure(figsize=(10, 3))
    plt.plot(df['SOC'], label='Battery SOC')
    plt.plot(df['EV_SOC'], label='EV SOC')
    plt.title("State of Charge")
    plt.ylabel("kWh")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True)

    # Binary state plots
    plt.figure(figsize=(10, 4))
    df[['u_CHP', 'u_DG', 'u_CH_ES', 'u_DIS_ES', 'A_EV']].astype(float).plot(drawstyle='steps-post')
    plt.title("Unit Status States")
    plt.ylabel("Binary State")
    plt.xlabel("Hour")
    plt.yticks([0, 1], ['Off', 'On'])
    plt.legend()
    plt.grid(True)

    # Startup indicator for DG
    plt.figure(figsize=(10, 2))
    df['startup_DG'].astype(float).plot(kind='bar', color='orange')
    plt.title("DG Startup Indicator")
    plt.ylabel("Startup Event")
    plt.xlabel("Hour")
    plt.grid(True)

    plt.show()