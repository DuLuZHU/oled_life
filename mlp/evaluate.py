import numpy as np


TARGET_NAMES = ['LT99', 'LT97', 'LT95', 'deltaV99', 'deltaV97', 'deltaV95', 'init_V']


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def evaluate_per_target(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = TARGET_NAMES
    
    metrics = {}
    for i, name in enumerate(target_names):
        metrics[name] = calculate_metrics(y_true[:, i], y_pred[:, i])
    
    return metrics


def print_metrics(metrics, title="评估指标"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'目标':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'R2':>10}")
    print(f"{'-'*60}")
    
    for target, m in metrics.items():
        print(f"{target:<12} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>9.2f}% {m['R2']:>10.4f}")


def enforce_monotonicity_LT(y_pred):
    LT_cols = [0, 1, 2]
    LT_pred = y_pred[:, LT_cols]
    LT_sorted = np.sort(LT_pred, axis=1)
    y_pred[:, LT_cols] = LT_sorted
    return y_pred


def enforce_monotonicity_dV(y_pred):
    dV_cols = [3, 4, 5]
    for i in range(1, 3):
        mask = y_pred[:, dV_cols[i]] < y_pred[:, dV_cols[i-1]]
        y_pred[mask, dV_cols[i]] = y_pred[mask, dV_cols[i-1]] + 1e-6
    return y_pred


def enforce_initV_nonnegative(y_pred):
    y_pred[:, 6] = np.maximum(y_pred[:, 6], 0.1)
    return y_pred


def apply_physical_constraints(y_pred):
    y_pred = enforce_monotonicity_LT(y_pred)
    y_pred = enforce_monotonicity_dV(y_pred)
    y_pred = enforce_initV_nonnegative(y_pred)
    return y_pred
