import os
import pickle
import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def load_pkl_data(data_dir):
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    
    X_materials = []
    X_conditions = []
    Y = []
    sample_names = []
    
    layers_order = ['anode', 'ht', 'em', 'et', 'cathod', 'cp']
    
    for f in pkl_files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        
        layer_features = [d[layer] for layer in layers_order]
        layers = np.stack(layer_features, axis=0)
        X_materials.append(layers)
        
        life = d['life']
        X_conditions.append(life[:2])
        Y.append(life[2:])
        
        sample_names.append(os.path.basename(f).replace('.pkl', ''))
    
    X_materials = np.array(X_materials)
    X_conditions = np.array(X_conditions)
    Y = np.array(Y)
    
    return X_materials, X_conditions, Y, sample_names


def flatten_features(X_materials, X_conditions):
    N = X_materials.shape[0]
    X_flat = X_materials.reshape(N, -1)
    X_full = np.concatenate([X_flat, X_conditions], axis=1)
    return X_full


def apply_log_transform(Y):
    return np.log1p(Y)


def inverse_log_transform(Y_log):
    return np.expm1(Y_log)


def split_data(X, Y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm, scaler


def normalize_targets(y_train, y_test):
    scaler = StandardScaler()
    y_train_norm = scaler.fit_transform(y_train)
    y_test_norm = scaler.transform(y_test)
    return y_train_norm, y_test_norm, scaler


def prepare_data(data_dir, use_log_transform=True, test_size=0.2, random_state=42):
    X_materials, X_conditions, Y, sample_names = load_pkl_data(data_dir)
    
    X_full = flatten_features(X_materials, X_conditions)
    
    if use_log_transform:
        Y_transformed = apply_log_transform(Y)
    else:
        Y_transformed = Y.copy()
    
    X_train, X_test, y_train, y_test = split_data(
        X_full, Y_transformed, test_size=test_size, random_state=random_state
    )
    
    X_train_norm, X_test_norm, scaler_X = normalize_features(X_train, X_test)
    y_train_norm, y_test_norm, scaler_y = normalize_targets(y_train, y_test)
    
    return {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train_norm,
        'y_test': y_test_norm,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'X_full': X_full,
        'Y': Y,
        'Y_transformed': Y_transformed,
        'X_materials': X_materials,
        'X_conditions': X_conditions,
        'sample_names': sample_names,
        'use_log_transform': use_log_transform,
        'input_dim': X_full.shape[1],
        'output_dim': Y.shape[1]
    }


def add_gaussian_noise(X, noise_std=0.01):
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X + noise


def compute_loss_weights(y_train, method='inverse_std'):
    if method == 'inverse_std':
        stds = y_train.std(axis=0)
        weights = 1.0 / (stds + 1e-8)
        weights = weights / weights.sum() * len(weights)
    elif method == 'uniform':
        weights = np.ones(y_train.shape[1])
    else:
        weights = np.ones(y_train.shape[1])
    return weights


def kfold_split(X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(X, y))
