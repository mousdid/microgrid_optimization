from solver import solve_model
from output import plot_results, save_report

def main():
    import gurobipy
    print(gurobipy.__version__)

    sol = solve_model()
    print("Optimal import profile:")
    for t, val in sol['Import'].items():
        print(f"t={t}, import={val}")
    
    # Save results to CSV
    save_report(sol, 'output/optimization_results.csv')
    
    # Plot visualization of results
    plot_results(sol)
    
    try:
        from convergence import plot_convergence
        plot_convergence('output/gurobi_log.txt')
    except Exception as e:
        print("Convergence plot skipped:", e)

if __name__ == '__main__': main()
