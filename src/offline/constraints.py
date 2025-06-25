from pyomo.environ import Constraint

def add_constraints(m):
    

    # Grid Import
    m.grid_import_lower = Constraint(m.T, rule=lambda m, t: m.p_import[t] >= 0)
    m.grid_import_upper = Constraint(m.T, rule=lambda m, t: m.p_import[t] <= m.P_grid_import_max[t])
    
    # Grid Export
    m.grid_export_lower = Constraint(m.T, rule=lambda m, t: m.p_export[t] >= 0)
    m.grid_export_upper = Constraint(m.T, rule=lambda m, t: m.p_export[t] <= m.P_grid_export_max[t])
    
    # # Wind Turbine
    m.wt_lower = Constraint(m.T, rule=lambda m, t: m.p_wt[t] >= 0)
    m.wt_upper = Constraint(m.T, rule=lambda m, t: m.p_wt[t] <= m.PWT_max[t])
    
    # # PV
    m.pv_lower = Constraint(m.T, rule=lambda m, t: m.p_pv[t] >= 0)
    m.pv_upper = Constraint(m.T, rule=lambda m, t: m.p_pv[t] <= m.PPV_max[t])
    
    # CHP with binaries
    m.chp_lower = Constraint(m.T, rule=lambda m, t: m.p_chp[t] >= 0)
    m.chp_upper = Constraint(m.T, rule=lambda m, t: m.p_chp[t] <= m.PCHP_max[t] * m.u_chp[t])
    
    # # DG with binaries
    m.dg_lower = Constraint(m.T, rule=lambda m, t: m.p_dg[t] >= 0)
    m.dg_upper = Constraint(m.T, rule=lambda m, t: m.p_dg[t] <= m.PDG_max[t] * m.u_dg[t])
    
    # # Storage charge
    m.ch_es_lower = Constraint(m.T, rule=lambda m, t: m.p_ch_es[t] >= 0)
    m.ch_es_upper = Constraint(m.T, rule=lambda m, t: m.p_ch_es[t] <= m.Pch_es_max[t] * m.u_ch_es[t])
    
    # # Storage discharge
    m.dis_es_lower = Constraint(m.T, rule=lambda m, t: m.p_dis_es[t] >= 0)
    m.dis_es_upper = Constraint(m.T, rule=lambda m, t: m.p_dis_es[t] <= m.Pdis_es_max[t] * m.u_dis_es[t])
    
    # EV charging limit
    m.ev_charge_lower = Constraint(m.T, rule=lambda m, t: m.p_ch_ev[t] >= 0)
    m.ev_charge_upper = Constraint(m.T, rule=lambda m, t: m.p_ch_ev[t] <= m.PEV_max[t] * m.A[t])
    
    # Mutually exclusive storage modes
    m.no_charge_discharge = Constraint(m.T, rule=lambda m, t: m.u_ch_es[t] + m.u_dis_es[t] <= 1)

    # Startup logic for DG
    def startup_dg_rule(m, t):
        if t == m.T.first():
            return m.e_startup_dg[t] >= m.u_dg[t]
        return m.e_startup_dg[t] >= m.u_dg[t] - m.u_dg[m.T.prev(t)]
    m.dg_startup = Constraint(m.T, rule=startup_dg_rule)

    # Startup logic for CHP
    def startup_chp_rule(m, t):
        if t == m.T.first():
            return m.e_startup_chp[t] >= m.u_chp[t]
        return m.e_startup_chp[t] >= m.u_chp[t] - m.u_chp[m.T.prev(t)]
    m.chp_startup = Constraint(m.T, rule=startup_chp_rule)


    # Heat production constraints
    m.heat_balance = Constraint(m.T, rule=lambda m, t: m.H_chp[t] == m.alpha_chp[t] * m.p_chp[t])
    m.heat_demand  = Constraint(m.T, rule=lambda m, t: m.H_demand[t] <= m.H_chp[t])

    #SOC dynamics for battery
    def soc_batt(m, t):
        prev = m.Ees_min[t] if t == m.T.first() else m.ees[m.T.prev(t)]
        return m.ees[t] == prev + m.eta_ch_es[t] * m.p_ch_es[t] - m.p_dis_es[t] / m.eta_dis_es[t]
    m.soc_batt = Constraint(m.T, rule=soc_batt)
    m.soc_min  = Constraint(m.T, rule=lambda m, t: m.ees[t] >= m.Ees_min[t])
    m.soc_max  = Constraint(m.T, rule=lambda m, t: m.ees[t] <= m.Ees_max[t])

    #SOC dynamics for EV + Ensure EV energy is 0 at session start
    def soc_ev(m, t):
        if t == m.T.first():
            return m.eev[t] == 0  # or any appropriate init value
        prev = m.T.prev(t)
        return m.eev[t] == (0 if m.session_start[t] == 1 else m.eev[prev]) + m.eta_ch_ev[t] * m.p_ch_ev[t]
    m.soc_ev = Constraint(m.T, rule=soc_ev)

# Ensure EV energy is within bounds
    def stop_ev_charging(m, t):
        return m.eev[t] <= 70  
    m.stop_ev_charging = Constraint(m.T, rule=stop_ev_charging)



    #Ensure EV energy is 0 at session start
    # def ev_session_start(m, t):
    #     if m.session_start[t] == 1:
    #         return m.eev[t] == 0
    #     return Constraint.Skip
    # m.ev_session_start = Constraint(m.T, rule=ev_session_start)

    #Ensure EV energy ≥ required at session end (leave)
    def ev_leave_requirement(m, t):
        if m.leave_possible[t] == 1:
            return m.eev[t] >= m.Eev_required[t]
        return Constraint.Skip
    m.ev_leave_requirement = Constraint(m.T, rule=ev_leave_requirement)

    # Power balance
    m.power_balance = Constraint(m.T, rule=lambda m, t:
        m.p_import[t]  + m.p_wt[t] + m.p_pv[t] + m.p_chp[t]+ m.p_dg[t] + m.p_dis_es[t] 
        == m.p_export[t] + m.param_load[t] +  m.p_ch_es[t] + m.p_ch_ev[t])

    return m
