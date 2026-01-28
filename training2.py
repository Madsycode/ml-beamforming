import argparse # Add this import at the top

# ... [Keep all imports, classes, and helper functions the same] ...

import argparse # Ensure this is imported

import os
import time
import numpy as np

# --- Non-interactive plotting backend first ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torchvision import transforms
try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    _has_new_api = True
except ImportError:
    from torchvision import models
    _has_new_api = False

# Speed up training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Dataset (Enhanced for normalization)
# -----------------------------
class BeamDataset(Dataset):
    def __init__(self, Ds, ws, transform=None):
        """
        Ds: Input Heatmaps
        ws: Target Beam Weights
        transform: Optional (required for ViT resizing)
        """
        # 1. Log-scale and Normalize Inputs (Crucial for convergence)
        # Avoid log(0) with 1e-9
        Ds_log = np.log10(Ds + 1e-9)
        
        # Min-Max Scale to [0, 1]
        self.min_val = np.min(Ds_log)
        self.max_val = np.max(Ds_log)
        Ds_norm = (Ds_log - self.min_val) / (self.max_val - self.min_val + 1e-8)
        
        self.Ds = torch.from_numpy(Ds_norm).float().unsqueeze(1) # (B, 1, S, S)
        
        ws_real = np.real(ws)
        ws_imag = np.imag(ws)
        self.ws = torch.from_numpy(np.concatenate([ws_real, ws_imag], axis=-1)).float()
        
        self.transform = transform

    def __len__(self):
        return len(self.Ds)

    def __getitem__(self, idx):
        d = self.Ds[idx]
        if self.transform is not None:
            d = self.transform(d)
        return d, self.ws[idx]

# -----------------------------
# Models
# -----------------------------

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

# -----------------------------
# Metrics & Utils
# -----------------------------

def alignment_loss(pred, gt, N):
    pred_real, pred_imag = pred[:, :N], pred[:, N:]
    gt_real, gt_imag = gt[:, :N], gt[:, N:]
    pred_c = torch.complex(pred_real, pred_imag)
    gt_c = torch.complex(gt_real, gt_imag)
    
    # Normalize prediction to ensure it's a valid beamforming vector direction
    pred_c = pred_c / (torch.linalg.vector_norm(pred_c, dim=1, keepdim=True) + 1e-8)
    
    # Loss = 1 - Correlation^2
    inner = torch.abs(torch.sum(torch.conj(pred_c) * gt_c, dim=1)) ** 2
    return (1.0 - inner).mean()

def load_dataset_h5(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with h5py.File(file_path, 'r') as f:
        Ds = np.array(f['Ds'])
        ws = np.array(f['ws'])
        S = int(f['global_params'].attrs['grid_size'])
        N = int(f['global_params'].attrs['N'])
    return Ds, ws, S, N

def get_loaders(ds_train, ds_val, ds_test, batch_size):
    # Use workers to parallelize the heavy ViT resizing if possible
    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    )

def train_engine(model, loaders, N, device, epochs=50, patience=15, lr=1e-4):
    train_loader, val_loader, _ = loaders
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = alignment_loss(pred, y, N)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += alignment_loss(pred, y, N).item()
        
        avg_val = val_loss / len(val_loader)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}", end='\r')
        
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            # Save best model state in memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
                
    print(f"\nTraining Complete. Best Val Loss: {best_loss:.4f}")
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    return history

# -----------------------------
# Main
# -----------------------------

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALL", choices=["CNN", "MLP", "ViT", "ALL"],
                        help="Select which model to train. Use specific names to run in parallel terminals.")
    parser.add_argument("--device", type=str, default="cuda", help="Force device (cuda/cpu)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("no cuda device found!")
        return

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 15
    LR = 1e-4

    # Device Setup
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on: {device}")

    # 2. Load Data (Common to all)
    try:
        Ds, ws, S, N = load_dataset_h5('datasets.h5')
        print(f"Data Loaded: {len(Ds)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Split Data Indices
    idx = np.arange(len(Ds))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=42)
    
    # Helper to subset data
    Ds_train, ws_train = Ds[train_idx], ws[train_idx]
    Ds_val, ws_val = Ds[val_idx], ws[val_idx]
    Ds_test, ws_test = Ds[test_idx], ws[test_idx]

    # 4. Define Execution List
    # We use a dictionary of functions (lambdas) to prevent instantiating 
    # models/loaders that we aren't going to use.
    models_to_run = []

    # --- CNN Configuration ---
    if args.model in ["CNN", "ALL"]:
        def setup_cnn():
            print("Setting up CNN...")
            loaders = get_loaders(
                BeamDataset(Ds_train, ws_train),
                BeamDataset(Ds_val, ws_val),
                BeamDataset(Ds_test, ws_test),
                BATCH_SIZE
            )
            model = CNNBeam(N).to(device)
            return "CNN", model, loaders
        models_to_run.append(setup_cnn)

    # --- MLP Configuration ---
    if args.model in ["MLP", "ALL"]:
        def setup_mlp():
            print("Setting up MLP...")
            loaders = get_loaders(
                BeamDataset(Ds_train, ws_train),
                BeamDataset(Ds_val, ws_val),
                BeamDataset(Ds_test, ws_test),
                BATCH_SIZE
            )
            model = MLPBeam(S*S, N).to(device)
            return "MLP", model, loaders
        models_to_run.append(setup_mlp)

    # --- ViT Configuration ---
    if args.model in ["ViT", "ALL"]:
        def setup_vit():
            print("Setting up ViT (GPU-based preprocessing)...")
            # Note: We now use the standard dataset (no CPU transform)
            # The resizing happens inside the ViTBeam model now.
            loaders = get_loaders(
                BeamDataset(Ds_train, ws_train), 
                BeamDataset(Ds_val, ws_val),
                BeamDataset(Ds_test, ws_test),
                BATCH_SIZE
            )
            model = ViTBeam(N, pretrained=True).to(device)
            return "ViT", model, loaders
        models_to_run.append(setup_vit)

    # 5. Execution Loop
    os.makedirs("figs", exist_ok=True)

    for setup_func in models_to_run:
        # Instantiate resources NOW (lazy loading)
        name, model, loaders = setup_func()
        
        print(f"\n==================================")
        print(f" STARTED TRAINING: {name}")
        print(f"==================================")
        
        t0 = time.time()
        history = train_engine(model, loaders, N, device, EPOCHS, PATIENCE, LR)
        duration = time.time() - t0
        
        # --- PLOT LOSS CURVE ---
        print(f"Plotting loss for {name}...")
        plt.figure(figsize=(10, 6))
        plt.plot(history['train'], label='Train Loss', linewidth=2)
        plt.plot(history['val'], label='Val Loss', linewidth=2)
        plt.title(f'{name} Training Dynamics')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (1 - Correlation^2)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_path = f"figures/{name}_loss.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")
        # -----------------------

        # Final Test Evaluation
        model.eval()
        test_loss = 0.0
        test_loader = loaders[2]
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_loss += alignment_loss(model(x), y, N).item()
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"FINISHED {name} | Test Loss: {avg_test_loss:.4f} | Time: {duration:.1f}s")

        # Save the model weights in case you want to load them later
        torch.save(model.state_dict(), f"models/{name}_model.pth")
        
        # Clean up GPU memory for the next model (if running ALL)
        del model
        del loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()