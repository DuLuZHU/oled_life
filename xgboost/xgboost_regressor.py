import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold


TARGET_CONFIGS = {
    'lnLT99': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.05,
        'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 1.5, 'reg_lambda': 2.0
    }, 'idx': 0},
    'lnLT97': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.05,
        'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 1.5, 'reg_lambda': 2.0
    }, 'idx': 1},
    'lnLT95': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.05,
        'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 1.5, 'reg_lambda': 2.0
    }, 'idx': 2},
    'dV99': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 4, 'n_estimators': 150, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 1.0
    }, 'idx': 3},
    'dV97': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 4, 'n_estimators': 150, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 1.0
    }, 'idx': 4},
    'dV95': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 4, 'n_estimators': 150, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 1.0
    }, 'idx': 5},
    'initV': {'loss': 'reg:squarederror', 'params': {
        'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.5
    }, 'idx': 6}
}


def train_xgboost_models(F_mol_train, cond_train, y_train, device_ids_train):
    models = {}
    best_n_estimators = {}
    
    X_train = np.concatenate([F_mol_train, cond_train], axis=1)
    
    for target_name, config in TARGET_CONFIGS.items():
        y_target = y_train[:, config['idx']]
        
        gkf = GroupKFold(n_splits=5)
        val_losses = []
        all_n_estimators = []
        
        for train_idx, val_idx in gkf.split(X_train, y_target, groups=device_ids_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_target[train_idx], y_target[val_idx]
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            params = {
                'objective': config['loss'],
                'eval_metric': 'rmse',
                'max_depth': config['params']['max_depth'],
                'learning_rate': config['params']['learning_rate'],
                'subsample': config['params']['subsample'],
                'colsample_bytree': config['params']['colsample_bytree'],
                'reg_alpha': config['params']['reg_alpha'],
                'reg_lambda': config['params']['reg_lambda'],
                'seed': 42
            }
            
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=config['params']['n_estimators'],
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                evals_result=evals_result,
                verbose_eval=False
            )
            
            best_iteration = model.best_iteration
            val_loss = evals_result['val']['rmse'][best_iteration]
            val_losses.append(val_loss)
            all_n_estimators.append(best_iteration)
        
        avg_val_loss = np.mean(val_losses)
        final_n_estimators = int(np.mean(all_n_estimators))
        
        print(f"{target_name}: 平均验证RMSE={avg_val_loss:.6f}, 最佳迭代次数={final_n_estimators}")
        
        dtrain_full = xgb.DMatrix(X_train, label=y_target)
        final_params = {
            'objective': config['loss'],
            'eval_metric': 'rmse',
            'max_depth': config['params']['max_depth'],
            'learning_rate': config['params']['learning_rate'],
            'subsample': config['params']['subsample'],
            'colsample_bytree': config['params']['colsample_bytree'],
            'reg_alpha': config['params']['reg_alpha'],
            'reg_lambda': config['params']['reg_lambda'],
            'seed': 42
        }
        
        final_model = xgb.train(final_params, dtrain_full, num_boost_round=final_n_estimators, verbose_eval=False)
        
        models[target_name] = final_model
        best_n_estimators[target_name] = final_n_estimators
    
    return models


def predict_xgboost(models, F_mol_data, cond_data):
    X_data = np.concatenate([F_mol_data, cond_data], axis=1)
    
    predictions = {}
    for target_name, model in models.items():
        dtest = xgb.DMatrix(X_data)
        pred = model.predict(dtest)
        predictions[target_name] = pred
    
    return predictions


def reverse_transform_predictions(predictions, dV_std, init_V_std):
    pred_lnLT99 = predictions['lnLT99']
    pred_lnLT97 = predictions['lnLT97']
    pred_lnLT95 = predictions['lnLT95']
    
    LT99 = np.expm1(pred_lnLT99)
    LT97 = np.expm1(pred_lnLT97)
    LT95 = np.expm1(pred_lnLT95)
    
    dV99 = predictions['dV99'] * dV_std[0]
    dV97 = predictions['dV97'] * dV_std[1]
    dV95 = predictions['dV95'] * dV_std[2]
    
    initV = predictions['initV'] * init_V_std
    
    return {
        'LT99': LT99,
        'LT97': LT97,
        'LT95': LT95,
        'deltaV99': dV99,
        'deltaV97': dV97,
        'deltaV95': dV95,
        'init_V': initV
    }
