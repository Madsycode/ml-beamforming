import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split 
import time

# Steering vector functions
def a_h(th, k0, d, N_h):
    return np.exp(1j * k0 * d * np.sin(th) * np.arange(N_h))

def a_v(ph, k0, d, N_v):
    return np.exp(1j * k0 * d * np.cos(ph) * np.arange(N_v))

def compute_R(D, THETA, PHI, k0, d, N_h, N_v):
    S = D.shape[0]
    N = N_h * N_v
    R = np.zeros((N, N), dtype=complex)
    for iy in range(S):
        for ix in range(S):
            th = THETA[iy, ix]
            ph = PHI[iy, ix]
            a = np.kron(a_v(ph, k0, d, N_v), a_h(th, k0, d, N_h))
            R += D[iy, ix] * np.outer(a, a.conj())
    return R

def compute_pattern(w, THETA, PHI, k0, d, N_h, N_v):
    S = THETA.shape[0]
    pattern = np.zeros((S, S))
    w = w / np.linalg.norm(w)
    for iy in range(S):
        for ix in range(S):
            th = THETA[iy, ix]
            ph = PHI[iy, ix]
            a = np.kron(a_v(ph, k0, d, N_v), a_h(th, k0, d, N_h))
            pattern[iy, ix] = np.abs(np.dot(w.conj(), a))**2
    return pattern

class BeamDataset(Dataset):
    def __init__(self, Ds, ws):
        self.Ds = torch.from_numpy(Ds).float().unsqueeze(1)  # (num_samples, 1, S, S)
        ws_real = np.real(ws)
        ws_imag = np.imag(ws)
        self.ws = torch.from_numpy(np.concatenate([ws_real, ws_imag], axis=-1)).float()  # (num_samples, 2*N)

    def __len__(self):
        return len(self.Ds)

    def __getitem__(self, idx):
        return self.Ds[idx], self.ws[idx]

class CNNBeam(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 4 * 4, 2 * N)  # 4096 to 512

    def forward(self, x):
        feat = self.conv(x)
        out = self.fc(feat)
        return out

def alignment_loss(pred, gt, N):
    # pred, gt: (B, 2*N)
    B = pred.shape[0]
    pred_real = pred[:, :N]
    pred_imag = pred[:, N:]
    pred_c = pred_real + 1j * pred_imag
    gt_real = gt[:, :N]
    gt_imag = gt[:, N:]
    gt_c = gt_real + 1j * gt_imag

    # Normalize pred
    norm_pred = torch.norm(pred_c, dim=1, keepdim=True)
    pred_c = pred_c / (norm_pred + 1e-8)  # Avoid div by zero

    # Assume gt is normalized
    inner = torch.abs(torch.sum(pred_c.conj() * gt_c, dim=1)) ** 2
    loss = 1 - inner
    return loss.mean()

def compute_mse(pred, gt, N, device):
    # pred, gt: (B, 2*N) on device
    B = pred.shape[0]
    pred_real = pred[:, :N]
    pred_imag = pred[:, N:]
    pred_c = (pred_real + 1j * pred_imag).to(torch.cdouble)
    gt_real = gt[:, :N]
    gt_imag = gt[:, N:]
    gt_c = (gt_real + 1j * gt_imag).to(torch.cdouble)

    # Normalize both
    norm_pred = torch.norm(pred_c, dim=1, keepdim=True)
    pred_c = pred_c / (norm_pred + 1e-8)
    norm_gt = torch.norm(gt_c, dim=1, keepdim=True)
    gt_c = gt_c / (norm_gt + 1e-8)

    # Align phases
    phase_diff = torch.angle(torch.sum(gt_c.conj() * pred_c, dim=1, keepdim=True))
    pred_c_aligned = pred_c * torch.exp(-1j * phase_diff)

    # Compute MSE
    diff = gt_c - pred_c_aligned
    mse = torch.mean(torch.abs(diff)**2, dim=1).mean()
    return mse

# Load dataset
with h5py.File('dataset.h5', 'r') as f:
    Ds = np.array(f['Ds'])
    ws = np.array(f['ws'])
    coverages = np.array(f['coverages'])
    params = f['params']
    S = params.attrs['S']
    N = params.attrs['N']
    theta_deg_range = params.attrs['theta_deg_range']
    phi_deg_range = params.attrs['phi_deg_range']
    lam = params.attrs['lambda']
    d = params.attrs['d']

k0 = 2 * np.pi / lam
N_h = 16  # Assuming fixed as per original
N_v = 16

# Angular mapping
theta = np.deg2rad((np.arange(S) - S/2) / (S/2) * theta_deg_range)
phi = np.deg2rad(np.arange(S) / S * phi_deg_range)
THETA, PHI = np.meshgrid(theta, phi)

# Split data
Ds_train, Ds_test, ws_train, ws_test, coverages_train, coverages_test = train_test_split(
    Ds, ws, coverages, test_size=0.01, random_state=42, shuffle=True)

train_dataset = BeamDataset(Ds_train, ws_train)
test_dataset = BeamDataset(Ds_test, ws_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model, optimizer, training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)} with CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available. Training on CPU.")
    
# Reload model for inference
model_path = 'model.pth'
loaded_model = CNNBeam(N).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print("Model reloaded for inference")

# Visualize for one test sample
idx = 550  # First test sample
D_vis = Ds_train[idx]
w_gt_vis = ws_train[idx]
R_vis = compute_R(D_vis, THETA, PHI, k0, d, N_h, N_v)

with torch.no_grad():
    D_i = train_dataset.Ds[idx].unsqueeze(0).to(device)
    pred = loaded_model(D_i)[0].cpu().numpy()
    pred_real = pred[:N]
    pred_imag = pred[N:]
    w_pred_vis = pred_real + 1j * pred_imag
    w_pred_vis /= np.linalg.norm(w_pred_vis)

pattern_gt = compute_pattern(w_gt_vis, THETA, PHI, k0, d, N_h, N_v)
pattern_pred = compute_pattern(w_pred_vis, THETA, PHI, k0, d, N_h, N_v)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(D_vis, cmap='hot', origin='lower')
axs[0].set_title('UE Demand Heatmap')
axs[1].imshow(pattern_gt, cmap='hot', origin='lower')
axs[1].set_title('Ground Truth Beam Pattern')
axs[2].imshow(pattern_pred, cmap='hot', origin='lower')
axs[2].set_title('Predicted Beam Pattern')
plt.show()