from solver import solve_model

def main():
    sol = solve_model()
    print("Optimal import profile:")
    for t,val in sol.items(): print(f"t={t}, import={val}")

if __name__ == '__main__': main()
