from pyomo.environ import ConcreteModel, Set, Param, Var, NonNegativeReals, Binary
from config import TIME_STEPS
from data_loader import load_parameters

def create_model():
    m = ConcreteModel()
    m.T = Set(initialize=TIME_STEPS, ordered=True)
    # load parameters
    raw = load_parameters()
    for pname, pvals in raw.items():
        setattr(m, pname, Param(m.T, initialize=pvals, mutable=False))
    # Power vars
    m.p_import = Var(m.T, domain=NonNegativeReals)
    m.p_export = Var(m.T, domain=NonNegativeReals)
    m.p_wt = Var(m.T, domain=NonNegativeReals)
    m.p_pv = Var(m.T, domain=NonNegativeReals)
    m.p_chp = Var(m.T, domain=NonNegativeReals)
    m.p_dg = Var(m.T, domain=NonNegativeReals)
    # Heat production from CHP
    m.H_chp = Var(m.T, domain=NonNegativeReals)
    # Unit-commitment binaries
    m.u_chp = Var(m.T, domain=Binary)
    m.u_dg = Var(m.T, domain=Binary)
   # m.e_startup_chp = Var(m.T, domain=Binary) #maybe to add
    m.e_startup_dg = Var(m.T, domain=Binary)
    # Storage vars & binaries
    m.p_ch_es = Var(m.T, domain=NonNegativeReals)
    m.p_dis_es = Var(m.T, domain=NonNegativeReals)
    m.u_ch_es = Var(m.T, domain=Binary)
    m.u_dis_es = Var(m.T, domain=Binary)
    m.ees = Var(m.T, domain=NonNegativeReals)
    # EV charging
    m.p_ch_ev = Var(m.T, domain=NonNegativeReals)
    # A_t binary variablity of a car
    m.A = Var(m.T, domain=Binary)
    return m
