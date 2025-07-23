from pyomo.environ import Objective, minimize


def add_objective(m):
    
    cost_import = sum(m.price_import[t]*m.p_import[t] for t in m.T)
    revenue_export = sum(m.price_export[t]*m.p_export[t] for t in m.T)
    cost_wt = sum(m.Cop_ma_wt[t] * m.p_wt[t] for t in m.T)
    cost_pv = sum(m.Cop_ma_pv[t] * m.p_pv[t] for t in m.T)
    fuel_chp = sum(m.rho_gas[t]*m.p_chp[t]/m.eta_chp[t] for t in m.T)
    fuel_dg = sum(m.rho_fuel[t]*m.p_dg[t]/m.eta_dg[t] for t in m.T)
    startup_chp = sum(m.C_startup[t]*m.e_startup_chp[t] for t in m.T) #maybe to add
    startup_dg = sum(m.C_startup[t]*m.e_startup_dg[t] for t in m.T)
    ev_charge_cost = sum(m.price_ev[t] * m.p_ch_ev[t] for t in m.T)
    C_degrad_es = sum(m.C_degrad_es[t] * m.p_dis_es[t] for t in m.T)  # degradation cost for storage discharge

    total_cost = cost_import - revenue_export - ev_charge_cost + cost_wt+ cost_pv+ fuel_chp+startup_chp + fuel_dg  + startup_dg+ C_degrad_es
    m.obj = Objective(expr=total_cost, sense=minimize)
    return m