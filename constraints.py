from pyomo.environ import Constraint, summation


def add_constraints(m):
    # Simple bounds
    m.grid_import_limit = Constraint(m.T, rule=lambda m,t: 0 <= m.p_import[t] <= m.P_grid_import_max[t])
    m.grid_export_limit = Constraint(m.T, rule=lambda m,t: 0 <= m.p_export[t] <= m.P_grid_export_max[t])
    m.wt_limit         = Constraint(m.T, rule=lambda m,t: 0 <= m.p_wt[t]   <= m.PWT_max[t])
    m.pv_limit         = Constraint(m.T, rule=lambda m,t: 0 <= m.p_pv[t]   <= m.PPV_max[t])
    # CHP & DG power bounds with binaries
    m.chp_limit = Constraint(m.T, rule=lambda m,t: 0 <= m.p_chp[t] <= m.PCHP_max[t]*m.u_chp[t])
    m.dg_limit  = Constraint(m.T, rule=lambda m,t: 0 <= m.p_dg[t]  <= m.PDG_max[t]*m.u_dg[t])
    # Storage charge/discharge bounds
    m.ch_es_limit  = Constraint(m.T, rule=lambda m,t: 0 <= m.p_ch_es[t] <= m.Pch_es_max[t]*m.u_ch_es[t])
    m.dis_es_limit= Constraint(m.T, rule=lambda m,t: 0 <= m.p_dis_es[t] <= m.Pdis_es_max[t]*m.u_dis_es[t])
    # EV charging bound
    m.ev_charge_limit = Constraint(m.T, rule=lambda m,t: 0 <= m.p_ch_ev[t] <= m.PEV_max[t]*m.A[t])

    # Mutually exclusive storage modes
    m.no_charge_discharge = Constraint(m.T, rule=lambda m,t: m.u_ch_es[t] + m.u_dis_es[t] <= 1)

# Startup logic for DG 
    def startup_dg_rule(m, t):
        if t == m.T.first():
            return m.e_startup_dg[t] >= m.u_dg[t]
        return m.e_startup_dg[t] >= m.u_dg[t] - m.u_dg[m.T.prev(t)]
    m.dg_startup = Constraint(m.T, rule=startup_dg_rule)
    
    # Heat production constraints
    m.heat_balance = Constraint(m.T, rule=lambda m,t: m.H_chp[t] == m.alpha_chp[t] * m.p_chp[t])
    m.heat_demand = Constraint(m.T, rule=lambda m,t: m.H_demand[t] <= m.H_chp[t])

    # SOC dynamics for battery
    def soc_batt(m, t):
        prev = 0 if t == m.T.first() else m.ees[m.T.prev(t)] #battery empty at the begining
        return m.ees[t] == prev + m.eta_ch_es[t]*m.p_ch_es[t] - m.p_dis_es[t]/m.eta_dis_es[t]
    m.soc_batt = Constraint(m.T, rule=soc_batt)
    m.soc_min  = Constraint(m.T, rule=lambda m,t: m.ees[t] >= m.Ees_min[t])
    m.soc_max  = Constraint(m.T, rule=lambda m,t: m.ees[t] <= m.Ees_max[t])

    # SOC dynamics for EV
    def soc_ev(m, t):
        prev = 0 if t == m.T.first() else m.Eev[m.T.prev(t)]
        return m.Eev[t] == prev + m.eta_ch_ev[t]*m.p_ch_ev[t]
    m.soc_ev = Constraint(m.T, rule=soc_ev)
    # Energy-at-departure requirement
    m.ev_final_req = Constraint(expr=summation(m.Eev_required) <= m.Eev[m.T.last()])

    #Power balance
    m.power_balance = Constraint(m.T, rule=lambda m,t:
        m.p_import[t] + m.p_dg[t] + m.p_chp[t] + m.p_wt[t] + m.p_pv[t] + m.p_dis_es[t]
        == m.p_export[t] + m.load[t] + m.p_ch_es[t] + m.p_ch_ev[t])

    return m