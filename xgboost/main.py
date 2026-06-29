import numpy as np
import json
import os
from data_utils import preprocess_data
from dnn_encoder import DNNCoder, train_encoder, extract_features
from xgboost_regressor import train_xgboost_models, predict_xgboost, reverse_transform_predictions
from post_processing import apply_all_constraints
from sklearn.model_selection import GroupShuffleSplit


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def main():
    data_dir = r'C:\Users\98179\Desktop\sanyue\new_model\lt_dataset'
    
    print("=" * 60)
    print("模块一：数据预处理")
    print("=" * 60)
    preprocessed = preprocess_data(data_dir)
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    cond_train = preprocessed['cond_train']
    cond_test = preprocessed['cond_test']
    dV_std = preprocessed['dV_std']
    init_V_std = preprocessed['init_V_std']
    life_data = preprocessed['life_data']
    device_ids = preprocessed['device_ids']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_train, groups=device_ids[:len(X_train)]))
    
    X_tr = X_train[train_idx]
    X_val = X_train[test_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[test_idx]
    device_ids_tr = [device_ids[i] for i in train_idx]
    device_ids_val = [device_ids[i] for i in test_idx]
    
    print(f"训练集样本数: {X_tr.shape[0]}")
    print(f"验证集样本数: {X_val.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print(f"特征维度: {X_tr.shape}")
    print(f"目标维度: {y_tr.shape}")
    
    print("\n" + "=" * 60)
    print("模块二：训练DNN编码器")
    print("=" * 60)
    encoder = DNNCoder()
    encoder = train_encoder(encoder, X_tr, y_tr, X_val, y_val)
    
    print("\n提取分子表征特征...")
    F_mol_train = extract_features(encoder, X_train)
    F_mol_test = extract_features(encoder, X_test)
    
    print(f"分子表征维度: {F_mol_train.shape}")
    
    print("\n" + "=" * 60)
    print("模块三：训练XGBoost回归器")
    print("=" * 60)
    train_device_ids = device_ids[:len(X_train)]
    xgb_models = train_xgboost_models(F_mol_train, cond_train, y_train, train_device_ids)
    
    print("\n预测测试集...")
    test_predictions = predict_xgboost(xgb_models, F_mol_test, cond_test)
    train_predictions = predict_xgboost(xgb_models, F_mol_train, cond_train)
    
    print("\n还原预测值...")
    test_pred_reversed = reverse_transform_predictions(test_predictions, dV_std, init_V_std)
    train_pred_reversed = reverse_transform_predictions(train_predictions, dV_std, init_V_std)
    
    print("\n" + "=" * 60)
    print("模块四：物理约束后处理")
    print("=" * 60)
    test_pred_constrained = apply_all_constraints(test_pred_reversed)
    train_pred_constrained = apply_all_constraints(train_pred_reversed)
    
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    test_life_data = life_data[len(X_train):]
    
    targets = ['LT99', 'LT97', 'LT95', 'deltaV99', 'deltaV97', 'deltaV95', 'init_V']
    target_indices = {'LT99': 2, 'LT97': 3, 'LT95': 4, 'deltaV99': 5, 'deltaV97': 6, 'deltaV95': 7, 'init_V': 8}
    
    print("\n训练集指标：")
    train_metrics = {}
    train_life_data = life_data[:len(X_train)]
    for target in targets:
        idx = target_indices[target]
        y_true = train_life_data[:, idx]
        y_pred = train_pred_constrained[target]
        metrics = calculate_metrics(y_true, y_pred)
        train_metrics[target] = metrics
        print(f"  {target}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2']:.4f}")
    
    print("\n测试集指标：")
    test_metrics = {}
    for target in targets:
        idx = target_indices[target]
        y_true = test_life_data[:, idx]
        y_pred = test_pred_constrained[target]
        metrics = calculate_metrics(y_true, y_pred)
        test_metrics[target] = metrics
        print(f"  {target}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2']:.4f}")
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'test_predictions': {k: v.tolist() for k, v in test_pred_constrained.items()}
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到 results/predictions.json")
    
    return results


if __name__ == '__main__':
    main()
