import re
import matplotlib.pyplot as plt

def parse_gurobi_log(log_file='gurobi_log.txt'):
    times, objs, bounds = [], [], []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'^\s*\d+\s+\d+\s+([\d\.]+)\s+([\d\.eE+\-]+)\s+([\d\.eE+\-]+)', line)
            if match:
                times.append(float(match.group(1)))
                objs.append(float(match.group(2)))
                bounds.append(float(match.group(3)))
    return times, objs, bounds

def plot_convergence(log_file='gurobi_log.txt'):
    times, objs, bounds = parse_gurobi_log(log_file)
    if not times:
        print("No MIP progress info found in log.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(times, objs, label='Best Objective')
    plt.plot(times, bounds, label='Best Bound')
    plt.xlabel("Time (s)")
    plt.ylabel("Objective Value")
    plt.title("Gurobi Convergence Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
