import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as param


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, use_sn=True):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        if use_sn:
            self.linear = param.spectral_norm(linear)
        else:
            self.linear = linear
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlockSN(nn.Module):
    def __init__(self, dim, dropout=0.2, use_sn=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        if use_sn:
            self.fc1 = param.spectral_norm(self.fc1)
            self.fc2 = param.spectral_norm(self.fc2)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)


class ResMLPSN(nn.Module):
    def __init__(self, input_dim, output_dim=7, hidden_dim=512, num_blocks=4, dropout=0.2, use_sn=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.input_proj = MLPBlock(input_dim, hidden_dim, dropout=dropout, use_sn=use_sn)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlockSN(hidden_dim, dropout=dropout, use_sn=use_sn) for _ in range(num_blocks)
        ])
        
        self.head = nn.Linear(hidden_dim, output_dim)
        if use_sn:
            self.head = param.spectral_norm(self.head)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


def mc_dropout_predict(model, X_data, n_samples=30, device=None, batch_size=256):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    model = enable_mc_dropout(model)
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    n = X_tensor.shape[0]
    output_dim = model.head.out_features if hasattr(model.head, 'out_features') else 7
    
    all_preds = torch.zeros(n_samples, n, output_dim, device=device)
    
    with torch.no_grad():
        for i in range(n_samples):
            preds = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = X_tensor[start:end]
                pred = model(batch)
                preds.append(pred)
            all_preds[i] = torch.cat(preds, dim=0)
    
    mean_pred = all_preds.mean(dim=0).cpu().numpy()
    std_pred = all_preds.std(dim=0).cpu().numpy()
    
    model = model.to('cpu')
    return mean_pred, std_pred
