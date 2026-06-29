import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy


def add_gaussian_noise(X, noise_std=0.01):
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X + noise


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32, shuffle=True):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
    
    def forward(self, pred, target):
        if self.weights is not None:
            w = self.weights.to(pred.device)
            loss = w * (pred - target) ** 2
        else:
            loss = (pred - target) ** 2
        return loss.mean()


def train_epoch(model, train_loader, criterion, optimizer, device, noise_std=0.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        if noise_std > 0:
            noise = torch.randn_like(X_batch) * noise_std
            X_batch = X_batch + noise
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=500, batch_size=32, lr=1e-3, weight_decay=1e-4,
                patience=50, device=None, loss_weights=None,
                noise_std=0.01, use_cosine_scheduler=True):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    criterion = WeightedMSELoss(
        weights=torch.tensor(loss_weights, dtype=torch.float32) if loss_weights is not None else None
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )
    
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, noise_std=noise_std)
        train_losses.append(train_loss)
        
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            if use_cosine_scheduler:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  早停触发，epoch={epoch+1}, 最佳val_loss={best_val_loss:.6f}")
                    break
        
        if (epoch + 1) % 50 == 0:
            val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}{val_str}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model = model.to('cpu')
    history = {'train_loss': train_losses, 'val_loss': val_losses, 'best_val_loss': best_val_loss}
    
    return model, history


def predict(model, X_data, device=None, batch_size=256):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())
    
    model = model.to('cpu')
    return np.concatenate(predictions, axis=0)


def kfold_cv(model_fn, X, y, n_splits=5, epochs=500, batch_size=32,
             lr=1e-3, weight_decay=1e-4, patience=50, device=None,
             loss_weights=None, noise_std=0.01, random_state=42):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_models = []
    oof_preds = np.zeros_like(y)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = model_fn()
        model, history = train_model(
            model, X_tr, y_tr, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            patience=patience, device=device, loss_weights=loss_weights,
            noise_std=noise_std
        )
        
        val_pred = predict(model, X_val, device=device)
        oof_preds[val_idx] = val_pred
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': history['best_val_loss'],
            'history': history
        })
        fold_models.append(model)
        
        print(f"  最佳验证loss: {history['best_val_loss']:.6f}")
    
    return fold_models, oof_preds, fold_results
