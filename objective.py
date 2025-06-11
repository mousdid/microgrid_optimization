from pyomo.environ import Objective, minimize


def add_objective(m):
    cost_import = sum(m.price_import[t]*m.p_import[t] for t in m.T)
    revenue_export = sum(m.price_export[t]*m.p_export[t] for t in m.T)
    fuel_chp = sum(m.rho_gas[t]*m.p_chp[t]/m.eta_chp[t] for t in m.T)
    fuel_dg = sum(m.rho_fuel[t]*m.p_dg[t]/m.eta_dg[t] for t in m.T)
    #startup_chp = sum(m.C_startup[t]*m.e_startup_chp[t] for t in m.T) #maybe to add
    startup_dg = sum(m.C_startup[t]*m.e_startup_dg[t] for t in m.T)
    total_cost = cost_import - revenue_export + fuel_chp + fuel_dg  + startup_dg
    m.obj = Objective(expr=total_cost, sense=minimize)
    return m