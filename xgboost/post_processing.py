import numpy as np


def apply_LT_constraint(LT99, LT97, LT95):
    LT99 = np.asarray(LT99)
    LT97 = np.asarray(LT97)
    LT95 = np.asarray(LT95)
    
    stacked = np.stack([LT99, LT97, LT95], axis=1)
    sorted_vals = np.sort(stacked, axis=1)
    
    LT99_out = sorted_vals[:, 0]
    LT97_out = sorted_vals[:, 1]
    LT95_out = sorted_vals[:, 2]
    
    return LT99_out, LT97_out, LT95_out


def apply_dV_constraint(dV99, dV97, dV95):
    dV99 = np.asarray(dV99)
    dV97 = np.asarray(dV97)
    dV95 = np.asarray(dV95)
    
    mask = dV99 > dV97
    dV97[mask] = (dV99[mask] + dV95[mask]) / 2.0
    
    mask2 = dV97 > dV95
    dV95[mask2] = dV97[mask2] + 1e-6
    
    return dV99, dV97, dV95


def apply_initV_constraint(init_V, min_init_V=0.1):
    init_V = np.asarray(init_V)
    init_V[init_V < 0] = min_init_V
    return init_V


def apply_all_constraints(predictions):
    LT99 = predictions['LT99']
    LT97 = predictions['LT97']
    LT95 = predictions['LT95']
    dV99 = predictions['deltaV99']
    dV97 = predictions['deltaV97']
    dV95 = predictions['deltaV95']
    init_V = predictions['init_V']
    
    LT99, LT97, LT95 = apply_LT_constraint(LT99, LT97, LT95)
    dV99, dV97, dV95 = apply_dV_constraint(dV99, dV97, dV95)
    init_V = apply_initV_constraint(init_V)
    
    return {
        'LT99': LT99,
        'LT97': LT97,
        'LT95': LT95,
        'deltaV99': dV99,
        'deltaV97': dV97,
        'deltaV95': dV95,
        'init_V': init_V
    }
