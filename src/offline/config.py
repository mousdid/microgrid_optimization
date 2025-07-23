# Time periods
HOURS_NUMBER = 720
TIME_STEPS = list(range(1, HOURS_NUMBER))  #to modify depending on data


DATA_DIR = "data/parameters/outage"


# Each CSV has one column with header matching the key
PARAM_FILES = {
    'param_load': 'load.csv',
    'price_import': 'price_import.csv',
    'price_export': 'price_export.csv',
    'rho_gas': 'rho_gas.csv',#maybe constant
    'eta_chp': 'eta_chp.csv',#maybe constant
    'Cop_ma_wt': 'Cop_ma_wt.csv',
    'Cop_ma_pv': 'Cop_ma_pv.csv',
    'rho_fuel': 'rho_fuel.csv',#maybe constant
    'eta_dg': 'eta_dg.csv',#maybe constant
    'C_startup': 'C_startup.csv',#maybe constant
    'C_degrad_es': 'C_degrad_es.csv',#maybe constant
    'price_ev': 'price_ev.csv',
    # Capacity limits
    'P_grid_import_max': 'P_grid_import_max.csv',
    'P_grid_export_max': 'P_grid_export_max.csv',
    'PWT_max': 'PWT_max.csv',
    'PPV_max': 'PPV_max.csv',
    'PCHP_max': 'PCHP_max.csv',
    'PDG_max': 'PDG_max.csv',
    'Pdis_es_max': 'Pdis_es_max.csv',
    'Pch_es_max': 'Pch_es_max.csv',
    'PEV_max': 'PEV_max.csv',
    # Storage and heat requirements
    'Ees_min': 'Ees_min.csv',
    'Ees_max': 'Ees_max.csv',
    'eta_ch_es': 'eta_ch_es.csv',#maybe constant
    'eta_dis_es': 'eta_dis_es.csv',#maybe constant
    'eta_ch_ev': 'eta_ch_ev.csv',#maybe constant
    'alpha_chp': 'alpha_chp.csv',#maybe constant
    'H_demand': 'H_demand.csv',
    'Eev_required': 'Eev_required.csv',
    # Ev car availability with start session and end sessions
     'A': 'A.csv',
     'session_start': 'session_start.csv',
     'leave_possible': 'leave_possible.csv',
}