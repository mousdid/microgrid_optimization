def update_battery_soc(prev_soc, p_charge, p_discharge, eta_charge, eta_discharge):
    """
    Update battery state of charge based on charge/discharge power
    
    Args:
        prev_soc: Previous state of charge
        p_charge: Charging power (kW)
        p_discharge: Discharging power (kW)
        eta_charge: Charging efficiency
        eta_discharge: Discharging efficiency
    
    Returns:
        Updated battery SOC
    """
    new_soc = prev_soc + eta_charge * p_charge - p_discharge / eta_discharge
    return new_soc

def update_ev_soc(prev_soc, p_charge, eta_charge, is_session_start):
    """
    Update EV state of charge based on charge power and session state
    
    Args:
        prev_soc: Previous state of charge
        p_charge: Charging power (kW)
        eta_charge: Charging efficiency
        is_session_start: Boolean indicating if this is the start of a charging session
    
    Returns:
        Updated EV SOC
    """
    if is_session_start:
        new_soc = 0 + eta_charge * p_charge
    else:
        new_soc = prev_soc + eta_charge * p_charge
    
    return new_soc

def process_grid_action(action, p_import_max, p_export_max):
    """
    Process grid action to determine import/export powers and grid mode
    
    Args:
        action: Grid control action in [-1, 1]
                Negative = export mode, Positive = import mode
        p_import_max: Maximum import power (kW)
        p_export_max: Maximum export power (kW)
    
    Returns:
        tuple: (p_import, p_export, u_maingrid)
               p_import: Import power (kW)
               p_export: Export power (kW) 
               u_maingrid: Grid mode binary (1=import, 0=export)
    """
    if action < 0:
        # Export mode
        p_import = 0.0
        p_export = abs(action) * p_export_max  # Convert negative action to positive export
        u_maingrid = 0
    else:
        # Import mode (including action = 0)
        p_import = action * p_import_max
        p_export = 0.0
        u_maingrid = 1
    
    return p_import, p_export, u_maingrid

def process_battery_action(action, p_ch_es_max, p_dis_es_max):
    """
    Process battery action to determine charge/discharge powers and battery mode
    
    Args:
        action: Battery control action in [-1, 1]
                Negative = discharge mode, Positive = charge mode
        p_ch_es_max: Maximum charging power (kW)
        p_dis_es_max: Maximum discharging power (kW)
    
    Returns:
        tuple: (p_ch_es, p_dis_es, u_es)
               p_ch_es: Charging power (kW)
               p_dis_es: Discharging power (kW)
               u_es: Battery mode binary (0=charging, 1=discharging)
    """
    if action < 0:
        # Discharge mode (negative action)
        p_ch_es = 0.0
        p_dis_es = abs(action) * p_dis_es_max
        u_es = 1
    else: 
        # Charge mode (positive action)
        p_ch_es = action * p_ch_es_max
        p_dis_es = 0.0
        u_es = 0
    
    return p_ch_es, p_dis_es, u_es
