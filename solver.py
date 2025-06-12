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
    solver.options['LogFile'] = 'output/gurobi_log.txt'
    results = solver.solve(m, tee=True,options={"DualReductions": 0})
    m.solutions.load_from(results)

    # Extracting the output
    report = {
      'Import':   {t: m.p_import[t].value for t in m.T},
      'Export':   {t: m.p_export[t].value for t in m.T},
      'PV':       {t: m.p_pv[t].value for t in m.T},
      'WT':       {t: m.p_wt[t].value for t in m.T},
      'DG':       {t: m.p_dg[t].value for t in m.T},
      'CHP':      {t: m.p_chp[t].value for t in m.T},
      'Heat_CHP': {t: m.H_chp[t].value for t in m.T},
      'Load_el':  {t: m.param_load[t] for t in m.T},
      'Load_th':  {t: m.H_demand[t] for t in m.T},
      'SOC':      {t: m.ees[t].value for t in m.T},
      'EV_SOC':   {t: m.eev[t].value for t in m.T},
      # Binaries
      'u_CHP':    {t: m.u_chp[t].value for t in m.T},
      'u_DG':     {t: m.u_dg[t].value for t in m.T},
      'u_CH_ES':  {t: m.u_ch_es[t].value for t in m.T},
      'u_DIS_ES': {t: m.u_dis_es[t].value for t in m.T},
      'A_EV':     {t: m.A[t] for t in m.T},
      'startup_DG':  {t: m.e_startup_dg[t].value for t in m.T},
    }

    return report

