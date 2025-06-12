from solver import solve_model

def main():
    
    import gurobipy
    print(gurobipy.__version__)


    sol = solve_model()
    print("Optimal import profile:")
    for t,val in sol.items(): print(f"t={t}, import={val}")
    try:
        from convergence import plot_convergence
        plot_convergence('gurobi_log.txt')
    except Exception as e:
        print("Convergence plot skipped:", e)
if __name__ == '__main__': main()
