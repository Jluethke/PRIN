"""
neuroprin/models.py

Neural network model definitions for NeuroPRIN.
Includes:
- DPLSTM (with custom DPLSTMCell)
- BaselineLSTM
- PRIN_LSTM (with attention, predictive coding, optional spiking)
- DPRIN_LSTM (with dynamic pruning)
- TemporalConvBlock
- FourierTransformLayer
- FeatureResonanceGating (v4 adaptive feature pruning)
- ChaosGating (v4 chaos-based scaling)
- TemporalAttention (v4 time-step pruning)
- RegimeEmbedding (v4 regime-aware embedding)
- NeuroPRINv4 (v4 core model)
- DirectionalMSELoss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


class FeatureResonanceGating(nn.Module):
    """Adaptive feature pruning via learned resonance gates."""
    def __init__(self, input_size: int, prune_rate: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.prune_rate = prune_rate
        self.gates = nn.Parameter(torch.ones(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gates

    def hard_prune(self) -> None:
        with torch.no_grad():
            threshold = torch.quantile(self.gates.abs(), self.prune_rate)
            mask = (self.gates.abs() >= threshold).float()
            self.gates.data *= mask


class ChaosGating(nn.Module):
    """Chaos-inspired gating to scale features by a chaos score."""
    def __init__(self, input_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(input_size))

    def forward(self, x, chaos_score):
        scale = 1 + 0.1 * chaos_score.unsqueeze(-1).unsqueeze(-1)
        return x * self.gate * scale



class TemporalAttention(nn.Module):
    """Soft attention weighting across time steps for dynamic pruning."""
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_weights = nn.Parameter(torch.ones(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.seq_weights, dim=0)
        return x * weights.unsqueeze(0).unsqueeze(-1)


class RegimeEmbedding(nn.Module):
    """Embed discrete regime states (from HMM) into a continuous space."""
    def __init__(self, num_regimes: int, embed_size: int):
        super().__init__()
        self.embed = nn.Embedding(num_regimes, embed_size)

    def forward(self, regimes: torch.LongTensor) -> torch.Tensor:
        return self.embed(regimes)


class DPLSTMCell(nn.Module):
    """Enhanced Adaptive Gated DPLSTM Cell."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)
        self.adaptive_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, hx: tuple):
        h, c = hx
        gates = self.W_ih(x) + self.W_hh(h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)
        i = torch.sigmoid(ingate)
        f = torch.sigmoid(forgetgate)
        g = torch.tanh(cellgate)
        o = torch.sigmoid(outgate)
        adaptive_strength = self.adaptive_gate(h)
        c_new = adaptive_strength * (f * c) + (1 - adaptive_strength) * (i * g)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class DPLSTM(nn.Module):
    """Stacked DPLSTM layers."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList([
            DPLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, hx: tuple = None):
        batch, seq_len, _ = x.size() if self.batch_first else (x.size(1), x.size(0), x.size(2))
        if hx is None:
            h = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
            c = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = list(hx[0]), list(hx[1])
        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :] if self.batch_first else x[t, :, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, (h[l], c[l]))
                inp = h[l]
            outputs.append(inp)
        out = torch.stack(outputs, dim=1 if self.batch_first else 0)
        h_n = torch.stack(h, dim=0).unsqueeze(0)
        c_n = torch.stack(c, dim=0).unsqueeze(0)
        return out, (h_n, c_n)


class BaselineLSTM(nn.Module):
    """Enhanced Baseline LSTM with residual connections and feed-forward refinement."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout_p: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p)
        self.residual_fc = nn.Linear(input_size, hidden_size)
        self.refine_fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        residual = self.residual_fc(x[:, -1, :])
        return self.refine_fc(out[:, -1, :] + residual)
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class SupportVectorModel:
    def __init__(self):
        self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X):
        return self.model.predict(X)


class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention with Adaptive Head Importance."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.head_scaling = nn.Parameter(torch.ones(num_heads))
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        head_imp = torch.softmax(self.head_scaling, dim=0).view(1, self.num_heads, 1, 1)
        attn = attn * head_imp
        out = (attn @ V).transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class PredictiveCodingLayer(nn.Module):
    """Hierarchical Predictive Coding Layer with feedback integration."""
    def __init__(self, dim: int, horizons: int = 3):
        super().__init__()
        self.pred_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(horizons)])
        self.feedback_layer = nn.Linear(dim*horizons, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = [layer(x) for layer in self.pred_layers]
        combined = torch.cat(preds, dim=-1)
        feedback = torch.tanh(self.feedback_layer(combined)).clamp(-5,5)
        return x - feedback


class SpikingLayer(nn.Module):
    """Adaptive Spiking Neuron Layer."""
    def __init__(self, input_dim: int, output_dim: int, initial_threshold: float=1.0, adaptation_rate: float=0.05):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.threshold = nn.Parameter(torch.full((output_dim,), initial_threshold))
        self.adaptation_rate = adaptation_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        spikes = []
        for t in range(x.size(1)):
            out = self.fc(x[:,t,:])
            mem += out
            s = (mem>=self.threshold).float()
            spikes.append(s.unsqueeze(1))
            self.threshold.data.add_(self.adaptation_rate*(s.mean(dim=0)-0.5).to(self.threshold.device))
            mem *= (1-s)
        return torch.cat(spikes, dim=1)


class DPRIN_LSTM(nn.Module):
    """Dynamic PRIN LSTM with RigL weight pruning/regrowth."""
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int=1, prune_rate:float=0.01, dropout_p:float=0.5, use_snn:bool=False):
        super().__init__()
        self.prune_rate = prune_rate
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p)
        self.attn = SelfAttention(hidden_size)
        self.pc = PredictiveCodingLayer(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = SpikingLayer(hidden_size, output_size) if use_snn else nn.Linear(hidden_size, output_size)

    def _rigl_prune_and_regrow(self, weight:torch.Tensor):
        flat = weight.abs().flatten()
        k = int(flat.numel()*self.prune_rate)
        if k<1: return
        thresh = torch.topk(flat,k,largest=False).values.max()
        mask = weight.abs()>=thresh
        pruned = weight*mask
        regrow = (~mask)&(torch.rand_like(weight)<self.prune_rate)
        pruned[regrow] = torch.randn_like(pruned[regrow])*0.01
        weight.data.copy_(pruned)

    def _apply_dynamic_pruning(self):
        for name,param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                self._rigl_prune_and_regrow(param.data)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        self._apply_dynamic_pruning()
        out,_ = self.lstm(x)
        att = self.attn(out)
        err = self.pc(att)
        combined = self.dropout(out+err)
        return self.fc(combined[:,-1,:])


class PRIN_LSTM(nn.Module):
    """PRIN LSTM with DPLSTM core, attention, predictive coding, optional spiking."""
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int=1, pruning_threshold:float=0.01, dropout_p:float=0.5, use_snn:bool=False):
        super().__init__()
        self.pruning_threshold = pruning_threshold
        self.lstm = DPLSTM(input_size, hidden_size, num_layers)
        self.attn = SelfAttention(hidden_size)
        self.pc = PredictiveCodingLayer(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = SpikingLayer(hidden_size, output_size) if use_snn else nn.Linear(hidden_size, output_size)
        self.hidden=None

    def reset_hidden(self,batch:int,device:torch.device):
        h0=torch.zeros(self.lstm.num_layers,batch,self.lstm.hidden_size,device=device)
        c0=torch.zeros_like(h0)
        self.hidden=(h0,c0)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if self.hidden is None or self.hidden[0].size(1)!=x.size(0):
            self.reset_hidden(x.size(0),x.device)
        out,self.hidden=self.lstm(x,self.hidden)
        mask=(out.abs()>self.pruning_threshold).float()
        out=out*mask
        att=self.attn(out)
        err=self.pc(att)
        combined=self.dropout(out+err)
        return self.fc(combined[:,-1,:])


class TemporalConvBlock(nn.Module):
    """Gated Residual Temporal Convolutional Block."""
    def __init__(self,in_ch:int,out_ch:int,kernel_size:int,dilation:int=1):
        super().__init__()
        padding=(kernel_size-1)*dilation//2
        self.conv_filter=nn.Conv1d(in_ch,out_ch,kernel_size,padding=padding,dilation=dilation)
        self.conv_gate=nn.Conv1d(in_ch,out_ch,kernel_size,padding=padding,dilation=dilation)
        self.dropout=nn.Dropout(0.2)
        self.residual=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        f=torch.tanh(self.conv_filter(x))
        g=torch.sigmoid(self.conv_gate(x))
        out=f*g
        out=self.dropout(out)
        return out+self.residual(x)


class FourierTransformLayer(nn.Module):
    """FFT-based feature expansion capturing magnitude & phase."""
    def __init__(self,use_fourier:bool=True,log_magnitude:bool=True):
        super().__init__()
        self.use_fourier=use_fourier
        self.log_magnitude=log_magnitude

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if not self.use_fourier:
            return x
        fft=torch.fft.rfft(x,dim=1)
        mag=torch.abs(fft)
        phase=torch.angle(fft)
        if self.log_magnitude:
            mag=torch.log1p(mag)
        return torch.cat([mag,phase],dim=-1)


class NeuroPRINv4(nn.Module):
    """NeuroPRIN v4 core model with multi-level adaptive pruning."""
    def __init__(self, input_size:int, seq_len:int, num_regimes:int, hidden_size:int=64, regime_embed_size:int=8, prune_rate:float=0.3):
        super().__init__()
        self.feature_gate = FeatureResonanceGating(input_size, prune_rate)
        self.chaos_gate = ChaosGating(input_size)
        self.temporal_attn = TemporalAttention(seq_len)
        self.regime_embed = RegimeEmbedding(num_regimes, regime_embed_size)
        self.lstm = nn.LSTM(input_size+regime_embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, 1)

    def forward(self, x:torch.Tensor, chaos_score:torch.Tensor, regime_state:torch.LongTensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.feature_gate(x)
        x = self.chaos_gate(x, chaos_score)
        x = self.temporal_attn(x)
        regimes = self.regime_embed(regime_state)
        regimes = regimes.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([x, regimes], dim=-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


class DirectionalMSELoss(nn.Module):
    """MSE loss weighted by directional accuracy."""
    def __init__(self, alpha:float=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        base = self.mse(pred, target)
        dir_acc = ((pred[1:]-pred[:-1])*(target[1:]-target[:-1])>0).float().mean()
        return base * (1 - self.alpha*dir_acc)
