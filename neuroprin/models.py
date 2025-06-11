"""
neuroprin/models.py

Neural network model definitions for NeuroPRIN.
Includes:
- DPLSTM (with custom DPLSTMCell)
- BaselineLSTM
- PRIN_LSTM (with attention, predictive coding, optional spiking)
- TemporalConvBlock
- FourierTransformLayer
- DirectionalMSELoss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPLSTMCell(nn.Module):
    """
    Enhanced Adaptive Gated DPLSTM Cell.
    Learns context-dependent recurrence strengths dynamically.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

        # Adaptive gating mechanism
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, hx: tuple):
        h, c = hx
        gates = self.W_ih(x) + self.W_hh(h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)
        i = torch.sigmoid(ingate)
        f = torch.sigmoid(forgetgate)
        g = torch.tanh(cellgate)
        o = torch.sigmoid(outgate)

        # Adaptive gating on forget/input interaction
        adaptive_strength = self.adaptive_gate(h)
        c_new = adaptive_strength * (f * c) + (1 - adaptive_strength) * (i * g)
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class DPLSTM(nn.Module):
    """
    Stacked DPLSTM layers.
    """
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
        # x: (batch, seq, feature) if batch_first
        if self.batch_first:
            batch, seq_len, _ = x.size()
        else:
            seq_len, batch, _ = x.size()
        # init states
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
    """
    Enhanced Baseline LSTM with residual connections and feed-forward refinement.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout_p: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_p
        )
        self.residual_fc = nn.Linear(input_size, hidden_size)  # residual path matching hidden size
        self.refine_fc = nn.Sequential(  # feed-forward refinement
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)

        # Residual connection from input projection
        residual = self.residual_fc(x[:, -1, :])

        out = out[:, -1, :] + residual

        # Refinement layer
        refined_output = self.refine_fc(out)

        return refined_output


class SelfAttention(nn.Module):
    """
    Self-attention mechanism.
    AdaptiveMultiHeadAttention
    Multi-Head Self-Attention with Adaptive Head Importance.
    Dynamically scales each head's influence based on learned signal strength.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Adaptive scaling parameters per head
        self.head_scaling = nn.Parameter(torch.ones(num_heads))
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Adaptive scaling per head (learned resonance per head)
        head_importance = torch.softmax(self.head_scaling, dim=0).view(1, self.num_heads, 1, 1)
        attn_weights = attn_weights * head_importance

        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(attn_output)




class PredictiveCodingLayer(nn.Module):
    """
    Predictive coding: learns to predict and computes error.
    HierarchicalPredictiveCodingLayer
    Hierarchical Predictive Coding Layer with feedback integration.
    Uses multiple prediction horizons for richer temporal understanding.
    """
    def __init__(self, dim: int, horizons: int = 3):
        super().__init__()
        self.horizons = horizons
        self.pred_layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(horizons)
        ])
        self.feedback_layer = nn.Linear(dim * horizons, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = [layer(x) for layer in self.pred_layers]

        # Combine multi-horizon predictions
        combined_preds = torch.cat(preds, dim=-1)
        feedback = torch.tanh(self.feedback_layer(combined_preds)).clamp(-5.0, 5.0)


        # Compute predictive coding error using hierarchical feedback
        error = x - feedback
        return error



class SpikingLayer(nn.Module):
    """
    Spiking Layer: Spiking neuron layer.    
    AdaptiveSpikingLayer
    Adaptive Spiking Neuron Layer.
    Dynamically adjusts firing thresholds based on recent activity.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_threshold: float = 1.0,
        adaptation_rate: float = 0.05
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.threshold = nn.Parameter(torch.full((output_dim,), initial_threshold))
        self.adaptation_rate = adaptation_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        spikes = []
        for t in range(x.size(1)):
            out = self.fc(x[:, t, :])
            mem += out
            current_spikes = (mem >= self.threshold).float()
            spikes.append(current_spikes.unsqueeze(1))
            # Threshold adaptation (Hebbian-like update)
            threshold_update = self.adaptation_rate * (current_spikes.mean(dim=0) - 0.5)
            self.threshold.data.add_(threshold_update.to(self.threshold.device))
            mem *= (1 - current_spikes)  # reset membrane potential on spike
        return torch.cat(spikes, dim=1)


class DPRIN_LSTM(nn.Module):
    """
    Dynamic PRIN LSTM: incorporates RigL-style weight pruning and regrowth
    on the recurrent and input-to-hidden weight matrices at each forward pass,
    combined with attention and predictive coding.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        prune_rate: float = 0.01,
        dropout_p: float = 0.5,
        use_snn: bool = False
    ):
        super().__init__()
        self.prune_rate = prune_rate
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_p
        )
        self.attn = SelfAttention(hidden_size)
        self.pc = PredictiveCodingLayer(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        if use_snn:
            self.fc = SpikingLayer(hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.use_snn = use_snn

    def _rigl_prune_and_regrow(self, weight: torch.Tensor) -> None:
        """
        In-place RigL pruning/regrowth on a weight tensor:
        - Prune the lowest-magnitude prune_rate fraction
        - Regrow random connections at the same fraction
        """
        # Compute threshold for pruning
        flat = weight.abs().flatten()
        k = int(flat.numel() * self.prune_rate)
        if k < 1:
            return
        threshold = torch.topk(flat, k, largest=False).values.max()
        # Prune
        mask = weight.abs() >= threshold
        pruned = weight * mask
        # Regrow
        regrow_mask = (~mask) & (torch.rand_like(weight) < self.prune_rate)
        pruned[regrow_mask] = torch.randn_like(pruned[regrow_mask]) * 0.01
        weight.data.copy_(pruned)

    def _apply_dynamic_pruning(self):
        """
        Apply RigL-style pruning/regrowth to all LSTM weight matrices.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                self._rigl_prune_and_regrow(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feature)
        # 1) Dynamic pruning/regrowth on LSTM weights
        self._apply_dynamic_pruning()

        # 2) Standard LSTM forward
        out, _ = self.lstm(x)

        # 3) Attention + predictive coding
        att = self.attn(out)
        err = self.pc(att)
        combined = self.dropout(out + err)

        # 4) Take last time-step and output
        last = combined[:, -1, :]
        return self.fc(last)


class PRIN_LSTM(nn.Module):
    """
    PRIN LSTM with pruning, attention, predictive coding, optional spiking output.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        pruning_threshold: float = 0.01,
        dropout_p: float = 0.5,
        use_snn: bool = False
    ):
        super().__init__()
        self.pruning_threshold = pruning_threshold
        self.lstm = DPLSTM(input_size, hidden_size, num_layers)
        self.attn = SelfAttention(hidden_size)
        self.pc = PredictiveCodingLayer(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        if use_snn:
            self.fc = SpikingLayer(hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        self.use_snn = use_snn
        self.hidden = None

    def reset_hidden(self, batch: int, device: torch.device) -> None:
        h0 = torch.zeros(self.lstm.num_layers, batch, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        self.hidden = (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feature)
        if self.hidden is None or self.hidden[0].size(1) != x.size(0):
            self.reset_hidden(x.size(0), x.device)
        out, self.hidden = self.lstm(x, self.hidden)
        # pruning
        mask = (out.abs() > self.pruning_threshold).float()
        out = out * mask
        # attention + predictive coding
        att = self.attn(out)
        err = self.pc(att)
        combined = self.dropout(out + err)
        # take last step
        last = combined[:, -1, :]
        return self.fc(last)


class TemporalConvBlock(nn.Module):
    """
    Residual temporal convolution block.
    GatedTemporalConvBlock
    Gated Residual Temporal Convolutional Block.
    Enhances temporal modeling by selectively gating information flow.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size,
                                     padding=padding, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=padding, dilation=dilation)

        self.dropout = nn.Dropout(0.2)

        # Adaptive residual connection
        self.residual = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq)
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        gated_out = filter_out * gate_out

        gated_out = self.dropout(gated_out)

        res = self.residual(x)

        return gated_out + res


class FourierTransformLayer(nn.Module):
    """
    FFT-based feature expansion.
    FourierMagnitudePhaseLayer
    Enhanced FFT-based layer capturing magnitude and phase explicitly.
    Useful for detecting resonant frequency patterns explicitly.
    """
    def __init__(self, use_fourier: bool = True, log_magnitude: bool = True):
        super().__init__()
        self.use_fourier = use_fourier
        self.log_magnitude = log_magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feature)
        if not self.use_fourier:
            return x

        fft = torch.fft.rfft(x, dim=1)
        
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)

        if self.log_magnitude:
            magnitude = torch.log1p(magnitude)

        return torch.cat([magnitude, phase], dim=-1)



class DirectionalMSELoss(nn.Module):
    """
    Enhanced Directional MSE Loss:
    Combines MSE, directional accuracy, adaptive variance regularization,
    and asymmetric penalty for predictions, optimizing for precise resonance inference.
    """
    def __init__(
        self,
        alpha: float = 0.01,
        epsilon: float = 1e-6,
        direction_weight: float = 0.1,
        asymmetry_factor: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.direction_weight = direction_weight
        self.asymmetry_factor = asymmetry_factor
        self.mse = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard MSE component
        mse_loss = self.mse(preds, targets)

        # Directional accuracy (encourages predicting correct direction of change)
        pred_diff = preds[1:] - preds[:-1]
        target_diff = targets[1:] - targets[:-1]

        pred_sign = torch.sign(pred_diff)
        target_sign = torch.sign(target_diff)

        directional_accuracy = (pred_sign == target_sign).float().mean()
        directional_penalty = (1 - directional_accuracy) * self.direction_weight

        # Adaptive variance regularization (based on recent error)
        pred_std = preds.std(unbiased=False)
        variance_reg = -self.alpha * torch.log(pred_std + self.epsilon)

        # Asymmetric penalty (heavier penalty for underestimations or overestimations)
        residuals = preds - targets
        asymmetric_penalty = torch.where(
            residuals > 0,
            residuals ** 2,
            self.asymmetry_factor * (residuals ** 2)
        ).mean()

        # Combine all components
        total_loss = mse_loss + directional_penalty + variance_reg + asymmetric_penalty

        return total_loss
