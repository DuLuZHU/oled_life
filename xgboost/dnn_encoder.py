import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DNNCoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(79, 16),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Dropout(0.3)
            ) for _ in range(6)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(176, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.4)
        )
        
        self.head_lnLT95 = nn.Linear(32, 1)
        self.head_initV = nn.Linear(32, 1)
    
    def encode(self, x):
        layer_embs = []
        for i in range(6):
            layer_feat = x[:, i, :]
            emb = self.layer_encoders[i](layer_feat)
            layer_embs.append(emb)
        
        anode, ht, em, et, cathod, cp = layer_embs
        
        diff_ht_anode = ht - anode
        diff_em_ht = em - ht
        diff_et_em = et - em
        diff_cath_et = cathod - et
        diff_cp_cath = cp - cathod
        
        concat = torch.cat([
            anode, ht, em, et, cathod, cp,
            diff_ht_anode, diff_em_ht, diff_et_em, diff_cath_et, diff_cp_cath
        ], dim=1)
        
        f_mol = self.fusion(concat)
        return f_mol
    
    def forward(self, x):
        f_mol = self.encode(x)
        pred_lnLT95 = self.head_lnLT95(f_mol)
        pred_initV = self.head_initV(f_mol)
        return pred_lnLT95, pred_initV


def train_encoder(model, X_train, y_train, X_val, y_val, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_lnLT95 = torch.tensor(y_train[:, 2:3], dtype=torch.float32).to(device)
    y_train_initV = torch.tensor(y_train[:, 6:7], dtype=torch.float32).to(device)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_lnLT95 = torch.tensor(y_val[:, 2:3], dtype=torch.float32).to(device)
    y_val_initV = torch.tensor(y_val[:, 6:7], dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    patience = 25
    counter = 0
    best_state_dict = None
    
    batch_size = 32
    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    
    for epoch in range(200):
        np.random.shuffle(indices)
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X_train_tensor[batch_idx]
            batch_lnLT95 = y_train_lnLT95[batch_idx]
            batch_initV = y_train_initV[batch_idx]
            
            optimizer.zero_grad()
            
            pred_lnLT95, pred_initV = model(batch_X)
            
            loss = 0.7 * criterion(pred_lnLT95, batch_lnLT95) + 0.3 * criterion(pred_initV, batch_initV)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        model.eval()
        with torch.no_grad():
            val_pred_lnLT95, val_pred_initV = model(X_val_tensor)
            val_loss = 0.7 * criterion(val_pred_lnLT95, y_val_lnLT95) + 0.3 * criterion(val_pred_initV, y_val_initV)
            val_loss = val_loss.item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_state_dict = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发，epoch={epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/200, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    model = model.to('cpu')
    return model


def extract_features(model, X_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        f_mol = model.encode(X_tensor)
    
    f_mol = f_mol.cpu().numpy()
    model = model.to('cpu')
    return f_mol
