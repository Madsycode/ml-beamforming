import os
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import torchvision stuff for ViT
try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _has_new_api = True
except ImportError:
    from torchvision import models
    _has_new_api = False

# ==========================================
# 1. DEFINE MODELS (Must match Training Script)
# ==========================================

class CNNBeam(nn.Module):
    """
    Fixed CNN with BatchNorm and Dropout to prevent early stopping/overfitting.
    """
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevent overfitting
            nn.Linear(1024, 2 * N)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class ViTBeam(nn.Module):
    def __init__(self, N, pretrained=True):
        super().__init__()
        # Load the base ViT model
        if _has_new_api:
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = models.vit_b_16(pretrained=pretrained)
        
        # Replace the classifier head to match our output size (2*N)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 2 * N)
        
        # Register ImageNet normalization constants as GPU buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x input shape is typically (Batch, 1, 128, 128)
        
        # 1. FORCE RESIZE to 224x224 (This fixes the AssertionError)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 2. Duplicate 1 channel to 3 channels (Grayscale -> RGB)
        x = x.repeat(1, 3, 1, 1)
        
        # 3. Apply standard ImageNet normalization
        x = (x - self.mean) / self.std
        
        # 4. Forward pass through the actual ViT
        return self.vit(x)
    
class MLPBeam(nn.Module):
    def __init__(self, input_size, N):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * N)
        )

    def forward(self, x):
        return self.fc(x)

# ==========================================
# 2. MATH UTILITIES
# ==========================================
def a_h(th, k0, d, N_h):
    return np.exp(1j * k0 * d * np.sin(th) * np.arange(N_h))

def a_v(ph, k0, d, N_v):
    return np.exp(1j * k0 * d * np.cos(ph) * np.arange(N_v))

def compute_pattern(w, THETA, PHI, k0, d, N_h, N_v):
    S = THETA.shape[0]
    pattern = np.zeros((S, S))
    w = w / (np.linalg.norm(w) + 1e-8)
    for iy in range(S):
        for ix in range(S):
            th = THETA[iy, ix]
            ph = PHI[iy, ix]
            a = np.kron(a_v(ph, k0, d, N_v), a_h(th, k0, d, N_h))
            pattern[iy, ix] = np.abs(np.dot(np.conj(w), a))**2
    return pattern

def precompute_steering_tables(THETA, PHI, k0, d, N_h, N_v):
    S = THETA.shape[0]
    idx_h = np.arange(N_h)[None, None, :]
    idx_v = np.arange(N_v)[None, None, :]
    ah = np.exp(1j * k0 * d * np.sin(THETA)[..., None] * idx_h)
    av = np.exp(1j * k0 * d * np.cos(PHI)[..., None] * idx_v)
    a = (ah[..., None, :] * av[..., :, None])
    a = a.reshape(S, S, N_h * N_v)
    return a

def Rx_noR(x, D, A_table):
    proj = np.tensordot(np.conj(x), A_table, axes=([0], [2]))
    y = np.tensordot(D * proj, A_table, axes=([0, 1], [0, 1]))
    return y

def power_iteration_noR(D, A_table, iters=10):
    N = A_table.shape[-1]
    x = np.random.randn(N) + 1j * np.random.randn(N)
    x = x / (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        y = Rx_noR(x, D, A_table)
        normy = np.linalg.norm(y) + 1e-12
        x = y / normy
    return x

# ==========================================
# 3. MAIN TEST LOGIC
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on: {device}")

    # --- Load Data ---
    print("Loading test dataset...")
    with h5py.File('datasets.h5', 'r') as f:
        Ds_raw = np.array(f['Ds'])
        ws_raw = np.array(f['ws'])
        
        # Load Params
        global_params = f['global_params']
        S = int(global_params.attrs['grid_size'])
        N = int(global_params.attrs['N'])
        N_h = int(global_params.attrs['N_h'])
        N_v = int(global_params.attrs['N_v'])
        lam = float(global_params.attrs['lambda'])
        d = float(global_params.attrs['d'])
        theta_range = float(global_params.attrs['theta_deg_range'])
        phi_range = float(global_params.attrs['phi_deg_range'])

    # --- Preprocessing (Must match training!) ---
    # Log Scale + Normalize
    Ds_log = np.log10(Ds_raw + 1e-9)
    min_val, max_val = np.min(Ds_log), np.max(Ds_log)
    Ds_norm = (Ds_log - min_val) / (max_val - min_val + 1e-8)
    
    # Split to get Test Set
    idx = np.arange(len(Ds_norm))
    _, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    
    Ds_test = Ds_norm[test_idx]
    ws_test = ws_raw[test_idx]
    Ds_test_raw_viz = Ds_raw[test_idx] # Keep raw for visualization

    # Convert to Tensor
    X_test = torch.from_numpy(Ds_test).float().unsqueeze(1) # (B, 1, S, S)
    y_test_real = torch.from_numpy(np.real(ws_test)).float()
    y_test_imag = torch.from_numpy(np.imag(ws_test)).float()
    Y_test = torch.cat([y_test_real, y_test_imag], dim=1)
    
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=1, shuffle=False)

    # --- Load Models ---
    models_dict = {}
    
    # Check if files exist
    if os.path.exists("models/MLP_model.pth"):
        m = MLPBeam(S*S, N).to(device)
        m.load_state_dict(torch.load("models/MLP_model.pth", map_location=device))
        models_dict["MLP"] = m.eval()

    if os.path.exists("models/CNN_model.pth"):
        m = CNNBeam(N).to(device)
        m.load_state_dict(torch.load("models/CNN_model.pth", map_location=device))
        models_dict["CNN"] = m.eval()
                
    if os.path.exists("models/ViT_model.pth"):
        m = ViTBeam(N).to(device)
        m.load_state_dict(torch.load("models/ViT_model.pth", map_location=device))
        models_dict["ViT"] = m.eval()

    if not models_dict:
        print("No .pth model files found in models/ folder! Run training first.")
        return

    # --- Precompute Geometry ---
    k0 = 2 * np.pi / lam
    theta = np.deg2rad((np.arange(S) - S / 2) / (S / 2) * theta_range)
    phi = np.deg2rad(np.arange(S) / S * phi_range)
    THETA, PHI = np.meshgrid(theta, phi)
    A_TABLE = precompute_steering_tables(THETA, PHI, k0, d, N_h, N_v)

    # ==========================================
    # 4. BENCHMARKING LOOP
    # ==========================================
    results = {'Model': [], 'Correlation': [], 'Time(ms)': []}
    
    # 1. Baseline Benchmark
    print("\nRunning Baseline (Power Iteration)...")
    t0 = time.time()
    corrs_base = []
    # Only run on first 50 samples to save time
    limit = min(50, len(Ds_test))
    for i in range(limit):
        # Raw demand used for baseline
        w_base = power_iteration_noR(Ds_test_raw_viz[i], A_TABLE, iters=10)
        
        # Calc Correlation with GT
        w_gt = ws_test[i]
        corr = np.abs(np.vdot(w_base, w_gt) / (np.linalg.norm(w_base)*np.linalg.norm(w_gt)))**2
        corrs_base.append(corr)
        
    base_time = (time.time() - t0) * 1000 / limit
    base_corr = np.mean(corrs_base)
    
    results['Model'].append('Baseline')
    results['Correlation'].append(base_corr)
    results['Time(ms)'].append(base_time)
    print(f"Baseline: Corr={base_corr:.4f}, Time={base_time:.2f}ms")

    # 2. ML Models Benchmark
    for name, model in models_dict.items():
        latencies = []
        correlations = []
        
        print(f"Testing {name}...")
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= limit: break
                x = x.to(device)
                
                # Timing
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_start = time.time()
                pred = model(x)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                latencies.append((time.time() - t_start) * 1000)
                
                # Accuracy
                pred_np = pred.cpu().numpy()[0]
                w_pred = pred_np[:N] + 1j * pred_np[N:]
                w_gt = y.numpy()[0, :N] + 1j * y.numpy()[0, N:]
                
                # Normalize
                w_pred /= (np.linalg.norm(w_pred) + 1e-9)
                w_gt /= (np.linalg.norm(w_gt) + 1e-9)
                
                corr = np.abs(np.vdot(w_pred, w_gt))**2
                correlations.append(corr)

        results['Model'].append(name)
        results['Correlation'].append(np.mean(correlations))
        results['Time(ms)'].append(np.mean(latencies))
        print(f"{name}: Corr={np.mean(correlations):.4f}, Time={np.mean(latencies):.2f}ms")

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    os.makedirs("figures", exist_ok=True)
    
    # Plot 1: Bar Chart Performance
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    models = results['Model']
    x = np.arange(len(models))
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Correlation (Higher is Better)', color='tab:blue')
    b1 = ax1.bar(x - 0.2, results['Correlation'], 0.4, label='Correlation', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Latency (ms) (Lower is Better)', color='tab:red')
    b2 = ax2.bar(x + 0.2, results['Time(ms)'], 0.4, label='Latency', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Accuracy vs Latency Comparison')
    plt.xticks(x, models)
    
    # Add legends
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/performance.png')
    print("Saved figures/performance.png")
    
    # Plot 2: Beam Patterns (Visual Inspection)
    # Pick a random sample index
    idx = 901 
    
    # GT Data
    D_in = Ds_test_raw_viz[idx]
    w_gt = ws_test[idx]
    pat_gt = compute_pattern(w_gt, THETA, PHI, k0, d, N_h, N_v)
    
    # Get Predictions
    pats = {}
    tensor_in = X_test[idx].unsqueeze(0).to(device)
    
    for name, model in models_dict.items():
        with torch.no_grad():
            pred = model(tensor_in).cpu().numpy()[0]
            w_p = pred[:N] + 1j * pred[N:]
            pats[name] = compute_pattern(w_p, THETA, PHI, k0, d, N_h, N_v)

    # Create Subplots
    num_plots = 2 + len(pats) # Input + GT + Models
    fig, axs = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    
    # Input
    im = axs[0].imshow(D_in, cmap='viridis', origin='lower')
    axs[0].set_title("Input Demand")
    #plt.colorbar(im, ax=axs[0], fraction=0.046)
    
    # GT
    im = axs[1].imshow(pat_gt, cmap='hot', origin='lower')
    axs[1].set_title("Ground Truth")
    #plt.colorbar(im, ax=axs[1], fraction=0.046)
    
    # Models
    for i, (name, pat) in enumerate(pats.items()):
        im = axs[2+i].imshow(pat, cmap='hot', origin='lower')
        axs[2+i].set_title(f"Pred: {name}")
        #plt.colorbar(im, ax=axs[2+i], fraction=0.046)
        
    plt.tight_layout()
    plt.savefig('figures/comparison.png')
    print("Saved figures/comparison.png")

if __name__ == "__main__":
    main()