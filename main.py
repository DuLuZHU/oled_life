import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_pkl_data, flatten_features, apply_log_transform,
    inverse_log_transform, normalize_features, normalize_targets,
    split_data, prepare_data, compute_loss_weights
)
from models import ResMLPSN, mc_dropout_predict
from train import kfold_cv, predict
from evaluate import evaluate_per_target, print_metrics, apply_physical_constraints, TARGET_NAMES


def ensemble_predict(models, X_data, device=None, n_mc_samples=30, use_mc_dropout=True):
    all_preds = []
    all_stds = []
    
    for model in models:
        if use_mc_dropout:
            mean_pred, std_pred = mc_dropout_predict(model, X_data, n_samples=n_mc_samples, device=device)
            all_preds.append(mean_pred)
            all_stds.append(std_pred)
        else:
            pred = predict(model, X_data, device=device)
            all_preds.append(pred)
    
    ensemble_mean = np.mean(all_preds, axis=0)
    
    if use_mc_dropout:
        within_model_var = np.mean(np.array(all_stds) ** 2, axis=0)
        between_model_var = np.var(all_preds, axis=0)
        ensemble_std = np.sqrt(within_model_var + between_model_var)
    else:
        ensemble_std = np.std(all_preds, axis=0)
    
    return ensemble_mean, ensemble_std


def main():
    data_dir = r'C:\Users\98179\Desktop\sanyue\new_model\lt_dataset'
    output_dir = r'C:\Users\98179\Desktop\sanyue\new_model\2.0\results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("OLED寿命预测模型 v2.0 (优化版)")
    print("=" * 60)
    print(f"使用设备: {device}")
    
    print("\n[1] 加载和预处理数据...")
    X_materials, X_conditions, Y, sample_names = load_pkl_data(data_dir)
    X_full = flatten_features(X_materials, X_conditions)
    Y_log = apply_log_transform(Y)
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(
        X_full, Y_log, test_size=0.2, random_state=42
    )
    
    X_train, X_test, scaler_X = normalize_features(X_train_raw, X_test_raw)
    y_train, y_test, scaler_y = normalize_targets(y_train_raw, y_test_raw)
    
    Y_train_true = Y[:len(X_train)]
    Y_test_true = Y[len(X_train):]
    
    loss_weights = compute_loss_weights(y_train, method='inverse_std')
    
    print(f"总样本数: {X_full.shape[0]}")
    print(f"训练集: {X_train.shape[0]}")
    print(f"测试集: {X_test.shape[0]}")
    print(f"输入维度: {X_train.shape[1]}")
    print(f"输出维度: {y_train.shape[1]}")
    print(f"损失权重: {loss_weights}")
    
    print("\n[2] 5折交叉验证训练...")
    def model_fn():
        return ResMLPSN(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            hidden_dim=512,
            num_blocks=4,
            dropout=0.2,
            use_sn=True
        )
    
    fold_models, oof_preds, fold_results = kfold_cv(
        model_fn, X_train, y_train,
        n_splits=5, epochs=500, batch_size=32,
        lr=1e-3, weight_decay=1e-4, patience=50,
        device=device, loss_weights=loss_weights,
        noise_std=0.01, random_state=42
    )
    
    oof_preds_orig = scaler_y.inverse_transform(oof_preds)
    oof_preds_orig = inverse_log_transform(oof_preds_orig)
    oof_metrics = evaluate_per_target(Y_train_true, oof_preds_orig)
    print_metrics(oof_metrics, "OOF验证集指标 (5折交叉验证)")
    
    print("\n[3] 集成模型预测测试集 (MC Dropout)...")
    test_pred_norm, test_std_norm = ensemble_predict(
        fold_models, X_test, device=device, n_mc_samples=30, use_mc_dropout=True
    )
    
    test_pred = scaler_y.inverse_transform(test_pred_norm)
    test_pred = inverse_log_transform(test_pred)
    test_std = test_std_norm * scaler_y.scale_
    
    test_pred_constrained = apply_physical_constraints(test_pred.copy())
    
    print("\n[4] 评估测试集性能...")
    test_metrics = evaluate_per_target(Y_test_true, test_pred_constrained)
    print_metrics(test_metrics, "测试集指标 (后处理后)")
    
    print("\n[5] 保存结果...")
    results = {
        'target_names': TARGET_NAMES,
        'oof_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in oof_metrics.items()},
        'test_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in test_metrics.items()},
        'fold_results': [{'fold': r['fold'], 'best_val_loss': float(r['best_val_loss'])} for r in fold_results],
        'test_predictions': test_pred_constrained.tolist(),
        'test_uncertainty': test_std.tolist(),
        'test_true': Y_test_true.tolist()
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    for i, model in enumerate(fold_models):
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_fold{i+1}.pth'))
    
    print(f"结果已保存到 {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
