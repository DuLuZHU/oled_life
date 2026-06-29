import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer


def load_pkl_files(data_dir):
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    data_list = []
    device_ids = []
    
    for pkl_file in pkl_files:
        file_path = os.path.join(data_dir, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        data_list.append(data)
        
        device_id = pkl_file.split('.')[0].rsplit('-', 1)[0]
        device_ids.append(device_id)
    
    return data_list, device_ids


def extract_features(data_list):
    layers = ['anode', 'ht', 'em', 'et', 'cathod', 'cp']
    features = []
    life_data = []
    conditions = []
    
    for data in data_list:
        layer_features = []
        for layer in layers:
            layer_data = data[layer]
            ecfp4 = layer_data[:1024]
            descriptors = layer_data[1024:]
            layer_features.append(np.concatenate([ecfp4, descriptors]))
        features.append(np.array(layer_features))
        
        life = data['life']
        life_data.append(life)
        
        J = life[0]
        T = life[1]
        log10J = np.log10(J) if J > 0 else 0
        T_K = T + 273.15
        inv_T = 1000.0 / T_K
        conditions.append([log10J, inv_T, log10J * inv_T])
    
    return np.array(features), np.array(life_data), np.array(conditions)


def apply_pca(features, n_components=64):
    num_samples, num_layers, num_features = features.shape
    
    pca_models = []
    pca_features = []
    
    for layer_idx in range(num_layers):
        layer_data = features[:, layer_idx, :1024]
        
        pca = IncrementalPCA(n_components=n_components)
        pca.fit(layer_data)
        
        pca_compressed = pca.transform(layer_data)
        descriptors = features[:, layer_idx, 1024:]
        
        combined = np.concatenate([pca_compressed, descriptors], axis=1)
        pca_features.append(combined)
        pca_models.append(pca)
    
    return np.stack(pca_features, axis=1), pca_models


def transform_targets(life_data):
    LT_cols = [2, 3, 4]
    dV_cols = [5, 6, 7]
    init_V_col = [8]
    
    y_LT = life_data[:, LT_cols]
    y_dV = life_data[:, dV_cols]
    y_init_V = life_data[:, init_V_col]
    
    y_logLT = np.log1p(y_LT)
    
    dV_std = y_dV.std(axis=0)
    y_dV_norm = y_dV / dV_std
    
    init_V_std = y_init_V.std()
    y_init_V_norm = y_init_V / init_V_std
    
    y_transformed = np.concatenate([y_logLT, y_dV_norm, y_init_V_norm], axis=1)
    
    return y_transformed, dV_std, init_V_std


def split_by_group(features, targets, conditions, device_ids, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(features, groups=device_ids))
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = targets[train_idx]
    y_test = targets[test_idx]
    cond_train = conditions[train_idx]
    cond_test = conditions[test_idx]
    
    return X_train, X_test, y_train, y_test, cond_train, cond_test


def normalize_features(X_train, X_test, cond_train, cond_test):
    flat_train = X_train.reshape(X_train.shape[0], -1)
    flat_train = np.concatenate([flat_train, cond_train], axis=1)
    
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    qt.fit(flat_train)
    
    flat_test = X_test.reshape(X_test.shape[0], -1)
    flat_test = np.concatenate([flat_test, cond_test], axis=1)
    
    X_train_norm_flat = qt.transform(flat_train)
    X_test_norm_flat = qt.transform(flat_test)
    
    num_layers = X_train.shape[1]
    num_layer_features = X_train.shape[2]
    num_cond = cond_train.shape[1]
    
    X_train_norm = X_train_norm_flat[:, :-num_cond].reshape(-1, num_layers, num_layer_features)
    cond_train_norm = X_train_norm_flat[:, -num_cond:]
    X_test_norm = X_test_norm_flat[:, :-num_cond].reshape(-1, num_layers, num_layer_features)
    cond_test_norm = X_test_norm_flat[:, -num_cond:]
    
    return X_train_norm, X_test_norm, cond_train_norm, cond_test_norm, qt


def preprocess_data(data_dir):
    data_list, device_ids = load_pkl_files(data_dir)
    features, life_data, conditions = extract_features(data_list)
    features_pca, pca_models = apply_pca(features)
    y_transformed, dV_std, init_V_std = transform_targets(life_data)
    
    X_train, X_test, y_train, y_test, cond_train, cond_test = split_by_group(
        features_pca, y_transformed, conditions, device_ids
    )
    
    X_train_norm, X_test_norm, cond_train_norm, cond_test_norm, qt = normalize_features(
        X_train, X_test, cond_train, cond_test
    )
    
    return {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train,
        'y_test': y_test,
        'cond_train': cond_train_norm,
        'cond_test': cond_test_norm,
        'pca_models': pca_models,
        'qt': qt,
        'dV_std': dV_std,
        'init_V_std': init_V_std,
        'life_data': life_data,
        'device_ids': device_ids
    }
