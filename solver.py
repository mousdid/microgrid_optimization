from pyomo.environ import SolverFactory
from model import create_model
from constraints import add_constraints
from objective import add_objective

def solve_model():
    m = create_model()
    m = add_constraints(m)
    m = add_objective(m)
    solver = SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.001
    results = solver.solve(m, tee=True)
    m.solutions.load_from(results)
    return {t: m.p_import[t].value for t in m.T}