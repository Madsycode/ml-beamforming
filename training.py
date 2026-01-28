# ============================
# Beamforming ML: CNN & MLP
# - Combined Visualization (Input, GT, CNN, MLP)
# - Non-blocking plotting
# - Fast evaluation
# ============================

import os
import time
import numpy as np

# --- Non-interactive plotting backend first ---
import matplotlib
matplotlib.use("Agg")  # prevents plt.show() from blocking
import matplotlib.pyplot as plt

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

# If available, enable some CUDA heuristics for speed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Steering vector utilities
# -----------------------------
def a_h(th, k0, d, N_h):
    return np.exp(1j * k0 * d * np.sin(th) * np.arange(N_h))

def a_v(ph, k0, d, N_v):
    return np.exp(1j * k0 * d * np.cos(ph) * np.arange(N_v))

def compute_pattern(w, THETA, PHI, k0, d, N_h, N_v):
    """Beam pattern on the SxS grid."""
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

def coverage_sinr_noR(w, D, A_table, sigma2=0.01):
    """Compute coverage and SINR without explicitly building R."""
    w = w / (np.linalg.norm(w) + 1e-8)
    proj = np.tensordot(np.conj(w), A_table, axes=([0], [2]))
    power = np.abs(proj)**2
    signal = float(np.sum(D * power).real)
    N = A_table.shape[-1]
    trace_R = float(N * np.sum(D))
    coverage = (signal / (trace_R + 1e-12)) * 100.0 if trace_R > 0 else 0.0
    sinr = 10.0 * np.log10(signal / (sigma2 + 1e-12)) if signal > 0 else -np.inf
    return coverage, sinr

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

# -----------------------------
# Dataset
# -----------------------------
class BeamDataset(Dataset):
    def __init__(self, Ds, ws):
        # 1. Normalize D: Log-scale usually works best for radio power maps
        # Adding 1e-9 to avoid log(0)
        Ds_log = np.log10(Ds + 1e-9)
        
        # 2. Min-Max Scale to range [0, 1] for stability
        self.min_val = np.min(Ds_log)
        self.max_val = np.max(Ds_log)
        Ds_norm = (Ds_log - self.min_val) / (self.max_val - self.min_val + 1e-8)
        
        self.Ds = torch.from_numpy(Ds_norm).float().unsqueeze(1)
        
        ws_real = np.real(ws)
        ws_imag = np.imag(ws)
        self.ws = torch.from_numpy(np.concatenate([ws_real, ws_imag], axis=-1)).float()

    def __len__(self):
        return len(self.Ds)

    def __getitem__(self, idx):
        return self.Ds[idx], self.ws[idx]
    
# class BeamDataset(Dataset):
#     def __init__(self, Ds, ws):
#         """
#         Ds: np.ndarray (num_samples, S, S)
#         ws: np.ndarray (num_samples, N) complex
#         """
#         self.Ds = torch.from_numpy(Ds).float().unsqueeze(1)  # (num, 1, S, S)
#         ws_real = np.real(ws)
#         ws_imag = np.imag(ws)
#         self.ws = torch.from_numpy(np.concatenate([ws_real, ws_imag], axis=-1)).float()  # (num, 2N)

#     def __len__(self):
#         return len(self.Ds)

#     def __getitem__(self, idx):
#         return self.Ds[idx], self.ws[idx]

# -----------------------------
# Models
# -----------------------------
class CNNBeam(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # <--- Added
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # <--- Added
            nn.ReLU(),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), # <--- Added
            nn.ReLU(),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), # <--- Added
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024), # Added an intermediate dense layer
            nn.ReLU(),
            nn.Dropout(0.3),              # Added dropout to prevent future overfitting
            nn.Linear(1024, 2 * N)
        )

    def forward(self, x):
        feat = self.conv(x)
        out = self.fc(feat)
        return out
    
# class CNNBeam(nn.Module):
#     def __init__(self, N):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
#             nn.AdaptiveAvgPool2d(4),
#             nn.Flatten()
#         )
#         self.fc = nn.Linear(256 * 4 * 4, 2 * N)

#     def forward(self, x):
#         return self.fc(self.conv(x))

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

# -----------------------------
# Losses & Metrics
# -----------------------------
def alignment_loss(pred, gt, N):
    """1 - |<w_pred, w_gt>|^2 with unit-norm pred."""
    pred_real, pred_imag = pred[:, :N], pred[:, N:]
    gt_real, gt_imag = gt[:, :N], gt[:, N:]

    pred_c = torch.complex(pred_real, pred_imag)
    gt_c = torch.complex(gt_real, gt_imag)

    pred_c = pred_c / (torch.linalg.vector_norm(pred_c, dim=1, keepdim=True) + 1e-8)

    inner = torch.abs(torch.sum(torch.conj(pred_c) * gt_c, dim=1)) ** 2
    loss = 1.0 - inner
    return loss.mean()

def compute_mse(pred, gt, N, device):
    """Phase-aligned MSE on normalized complex vectors."""
    pred_c = torch.complex(pred[:, :N], pred[:, N:])
    gt_c = torch.complex(gt[:, :N], gt[:, N:])

    pred_c = pred_c / (torch.linalg.vector_norm(pred_c, dim=1, keepdim=True) + 1e-8)
    gt_c = gt_c / (torch.linalg.vector_norm(gt_c, dim=1, keepdim=True) + 1e-8)

    inner = torch.sum(torch.conj(gt_c) * pred_c, dim=1, keepdim=True)
    phase = torch.angle(inner)
    pred_aligned = pred_c * torch.exp(-1j * phase)

    return torch.mean(torch.abs(gt_c - pred_aligned) ** 2)

def compute_phase_mae(pred_np, gt_np, N):
    pred = torch.tensor(pred_np)
    gt = torch.tensor(gt_np)
    pred_c = torch.complex(pred[:, :N], pred[:, N:])
    gt_c = torch.complex(gt[:, :N], gt[:, N:])
    
    diff = torch.abs(torch.angle(pred_c) - torch.angle(gt_c))
    mae = torch.min(diff, 2 * np.pi - diff).mean()
    return mae.item()

# -----------------------------
# Data loading
# -----------------------------
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found.")

    with h5py.File(file_path, 'r') as f:
        Ds = np.array(f['Ds'])
        ws = np.array(f['ws'])
        coverages = np.array(f['coverages'])
        
        global_params = f['global_params']
        S = int(global_params.attrs['grid_size'])
        N = int(global_params.attrs['N'])
        N_h = int(global_params.attrs['N_h'])
        N_v = int(global_params.attrs['N_v'])
        theta_deg_range = float(global_params.attrs['theta_deg_range'])
        phi_deg_range = float(global_params.attrs['phi_deg_range'])
        lam = float(global_params.attrs['lambda'])
        d = float(global_params.attrs['d'])

        param_grp = f['sample_params']
        params_list = []
        for i in range(len(Ds)):
            grp = param_grp[str(i)]
            p = {}
            for k in grp.attrs:
                val = grp.attrs[k]
                if isinstance(val, (bytes, str)):
                    val = val.decode() if isinstance(val, bytes) else val
                p[k] = val
            params_list.append(p)

    k0 = 2 * np.pi / lam
    theta = np.deg2rad((np.arange(S) - S / 2) / (S / 2) * theta_deg_range)
    phi = np.deg2rad(np.arange(S) / S * phi_deg_range)
    THETA, PHI = np.meshgrid(theta, phi)

    return Ds, ws, coverages, params_list, S, N, N_h, N_v, k0, d, THETA, PHI

# -----------------------------
# Training / Evaluation
# -----------------------------
def get_loaders(dataset_train, dataset_val, dataset_test, batch_size=32):
    num_workers = 2 if os.name != 'nt' else 0
    pin = torch.cuda.is_available()
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=num_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                            pin_memory=pin, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                             pin_memory=pin, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, N, device, num_epochs=100, patience=10):
    optimizer = Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    train_losses = []
    val_losses = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = len(train_loader)
        
        for i, (D_batch, w_batch) in enumerate(train_loader):
            D_batch = D_batch.to(device, non_blocking=True)
            w_batch = w_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(D_batch)
            loss = alignment_loss(pred, w_batch, N)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())

            if i % 20 == 0 and i > 0:
                print(f"  Batch {i}/{num_batches}...", end='\r')

        avg_train_loss = total_train_loss / max(1, num_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for D_batch, w_batch in val_loader:
                D_batch = D_batch.to(device, non_blocking=True)
                w_batch = w_batch.to(device, non_blocking=True)
                pred = model(D_batch)
                val_loss = alignment_loss(pred, w_batch, N)
                total_val_loss += float(val_loss.item())
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    return model, train_losses, val_losses

def evaluate_model_fast(model, loader, params_subset, Ds_subset, ws_subset,
                        coverages_subset, N, A_table, device, sigma2=0.01):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_coverage = []
    total_sinr = []
    total_baseline_coverage = []
    total_baseline_sinr = []

    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (D_batch, w_batch) in enumerate(loader):
            D_batch = D_batch.to(device, non_blocking=True)
            w_batch = w_batch.to(device, non_blocking=True)
            pred = model(D_batch)

            mse = compute_mse(pred, w_batch, N, device)
            mae = compute_phase_mae(pred.cpu().numpy(), w_batch.cpu().numpy(), N)
            total_mse += float(mse.item())
            total_mae += float(mae)

            B = D_batch.shape[0]
            for j in range(B):
                sample_idx = batch_idx * loader.batch_size + j
                if sample_idx >= len(Ds_subset):
                    break
                D_i = Ds_subset[sample_idx]
                w_gt_i = ws_subset[sample_idx]

                pred_i = pred[j].detach().cpu().numpy()
                w_pred = pred_i[:N] + 1j * pred_i[N:]
                w_pred = w_pred / (np.linalg.norm(w_pred) + 1e-8)

                cov_pred, sinr_pred = coverage_sinr_noR(w_pred, D_i, A_table, sigma2)
                total_coverage.append(cov_pred)
                total_sinr.append(sinr_pred)

                # Baseline from GT
                total_baseline_coverage.append(coverages_subset[sample_idx])
                _, sinr_gt = coverage_sinr_noR(w_gt_i, D_i, A_table, sigma2)
                total_baseline_sinr.append(sinr_gt)

    avg_mse = total_mse / max(1, num_batches)
    avg_mae = total_mae / max(1, num_batches)
    avg_coverage = float(np.mean(total_coverage)) if total_coverage else 0.0
    avg_sinr = float(np.mean(total_sinr)) if total_sinr else 0.0
    avg_baseline_coverage = float(np.mean(total_baseline_coverage)) if total_baseline_coverage else 0.0
    avg_baseline_sinr = float(np.mean(total_baseline_sinr)) if total_baseline_sinr else 0.0

    return avg_mse, avg_mae, avg_coverage, avg_sinr, avg_baseline_coverage, avg_baseline_sinr


# -----------------------------
# Main
# -----------------------------
def main():
    DO_PLOTS = True
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    PATIENCE = 10
    SIGMA2 = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    try:
        Ds, ws, coverages, params_list, S, N, N_h, N_v, k0, d, THETA, PHI = load_dataset('datasets.h5')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loaded dataset: {len(Ds)} samples | Grid {S}x{S} | N={N}")

    # Split indices
    indices = np.arange(len(Ds))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=1/9, random_state=42)

    Ds_train, ws_train = Ds[train_idx], ws[train_idx]
    Ds_val, ws_val = Ds[val_idx], ws[val_idx]
    Ds_test, ws_test = Ds[test_idx], ws[test_idx]

    coverages_test = coverages[test_idx]
    params_test = [params_list[i] for i in test_idx]

    # Precompute A_TABLE for metrics
    print("Precomputing steering table A_TABLE...")
    A_TABLE = precompute_steering_tables(THETA, PHI, k0, d, N_h, N_v)

    print("Preparing datasets and loaders...")
    train_dataset = BeamDataset(Ds_train, ws_train)
    val_dataset = BeamDataset(Ds_val, ws_val)
    test_dataset = BeamDataset(Ds_test, ws_test)

    # Dictionary of models to train
    models_dict = {
        'CNN': CNNBeam(N).to(device),
        'MLP': MLPBeam(S * S, N).to(device),
    }

    # Common loaders
    loaders = get_loaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE)

    results = {}
    os.makedirs("figs", exist_ok=True)

    for model_name, model in models_dict.items():
        print(f"\n=== Training {model_name} ===")
        # loaders is tuple (train, val, test)
        train_loader, val_loader, test_loader = loaders

        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, N, device,
            num_epochs=NUM_EPOCHS, patience=PATIENCE
        )

        # Plot losses
        if DO_PLOTS:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Training and Validation Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"figs/{model_name}_loss.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Evaluate
        print(f"Evaluating {model_name} on test set...")
        t0 = time.time()
        avg_mse, avg_mae, avg_coverage, avg_sinr, _, _ = evaluate_model_fast(
            model, test_loader, params_test, Ds_test, ws_test, coverages_test,
            N, A_TABLE, device, sigma2=SIGMA2
        )
        eval_time = time.time() - t0
        print(f"{model_name} evaluation done in {eval_time:.2f}s")

        # Inference Timing
        infer_times = []
        with torch.no_grad():
            for i, (D_batch, _) in enumerate(test_loader):
                if i > 10: break 
                D_batch = D_batch.to(device, non_blocking=True)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start = time.time()
                _ = model(D_batch)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                per_sample_ms = (time.time() - start) * 1000 / D_batch.shape[0]
                infer_times.append(per_sample_ms)
        avg_infer = float(np.mean(infer_times)) if infer_times else 0.0

        results[model_name] = {
            'MSE': avg_mse,
            'MAE (rad)': avg_mae,
            'Coverage (%)': avg_coverage,
            'SINR (dB)': avg_sinr,
            'Inference (ms)': avg_infer,
        }

    # Baseline (Power Iteration)
    print("\nTiming baseline (power iteration, no R build)...")
    baseline_times = []
    baseline_sinrs = []
    limit_baseline = min(len(Ds_test), 200)  # Limit for speed
    for i in range(limit_baseline):
        start = time.time()
        _ = power_iteration_noR(Ds_test[i], A_TABLE, iters=10)
        baseline_times.append((time.time() - start) * 1000.0)
        _, sinr_gt = coverage_sinr_noR(ws_test[i], Ds_test[i], A_TABLE, sigma2=SIGMA2)
        baseline_sinrs.append(sinr_gt)

    avg_baseline_time = float(np.mean(baseline_times)) if baseline_times else 0.0
    avg_baseline_coverage = float(np.mean(coverages_test[:limit_baseline]))
    avg_baseline_sinr = float(np.mean(baseline_sinrs)) if baseline_sinrs else 0.0

    # Print Table
    print("\nQuantitative Metrics: Baseline vs. CNN/MLP")
    print("| Model    | MSE        | MAE (rad) | Coverage (%) | SINR (dB) | Inference (ms) |")
    print("|----------|------------|-----------|--------------|-----------|----------------|")
    print(f"| Baseline | -          | -         | {avg_baseline_coverage:>6.1f}       | {avg_baseline_sinr:>6.1f}   | {avg_baseline_time:>12.1f} |")
    for model_name, res in results.items():
        print(f"| {model_name:<8}| {res['MSE']:.2e} | {res['MAE (rad)']:.2f}    | "
              f"{res['Coverage (%)']:.1f}         | {res['SINR (dB)']:.1f}    | {res['Inference (ms)']:.1f}         |")

    # ---------------------------------------------------------
    # Combined Visualizations (CNN, MLP, GT, Input)
    # ---------------------------------------------------------
    if len(Ds_test) > 0 and 'CNN' in models_dict and 'MLP' in models_dict:
        print("\nGenerating combined visualization (Input, GT, CNN, MLP)...")
        
        # Select first test sample
        idx = 0 
        D_vis = Ds_test[idx]
        w_gt = ws_test[idx]
        
        # Prepare Tensor
        D_tensor = test_dataset.Ds[idx].unsqueeze(0).to(device) # (1, 1, S, S)
        
        # 1. CNN Prediction
        model_cnn = models_dict['CNN']
        model_cnn.eval()
        with torch.no_grad():
            pred_cnn = model_cnn(D_tensor)[0].cpu().numpy()
            w_cnn = pred_cnn[:N] + 1j * pred_cnn[N:]
            w_cnn /= (np.linalg.norm(w_cnn) + 1e-8)
            
        # 2. MLP Prediction
        model_mlp = models_dict['MLP']
        model_mlp.eval()
        with torch.no_grad():
            pred_mlp = model_mlp(D_tensor)[0].cpu().numpy()
            w_mlp = pred_mlp[:N] + 1j * pred_mlp[N:]
            w_mlp /= (np.linalg.norm(w_mlp) + 1e-8)
            
        # 3. Compute Patterns
        pattern_gt = compute_pattern(w_gt, THETA, PHI, k0, d, N_h, N_v)
        pattern_cnn = compute_pattern(w_cnn, THETA, PHI, k0, d, N_h, N_v)
        pattern_mlp = compute_pattern(w_mlp, THETA, PHI, k0, d, N_h, N_v)
        
        # 4. Plot
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input Heatmap
        im0 = axs[0].imshow(D_vis, cmap='viridis', origin='lower')
        axs[0].set_title('Input: UE Demand')
        axs[0].scatter(64, 0, marker='^', color='red', s=200, label='BS')
        #plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        # GT Pattern
        im1 = axs[1].imshow(pattern_gt, cmap='hot', origin='lower')
        axs[1].set_title('Target: Ground Truth')
        #plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        
        # CNN Pattern
        im2 = axs[2].imshow(pattern_cnn, cmap='hot', origin='lower')
        axs[2].set_title('Pred: CNN')
        #plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        
        # MLP Pattern
        im3 = axs[3].imshow(pattern_mlp, cmap='hot', origin='lower')
        axs[3].set_title('Pred: MLP')
        #plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig("figs/prediction.png", dpi=150, bbox_inches="tight")
        plt.close()
            
    print("\nAll done. Figures saved to ./figs")

if __name__ == "__main__":
    main()