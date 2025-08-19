import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import argparse

# Set a random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Data Handling: PyTorch Dataset and Loaders ---

class MIMODataset(Dataset):
    """Custom PyTorch Dataset for loading MIMO heatmap and weight data."""
    def __init__(self, json_path: Path, fallback_antennas=16, fallback_res=128):
        self.params = {}
        if not json_path.exists():
            print(f"Warning: '{json_path}' not found. Creating a small dummy dataset.")
            create_dummy_dataset(json_path, fallback_antennas, fallback_res)
        
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if self.data:
            # Infer parameters from the first sample in the dataset
            first_sample_params = self.data[0].get('params', {})
            heatmap_shape = np.array(self.data[0]['heatmap']).shape
            self.params['num_antennas'] = first_sample_params.get('num_antennas', len(self.data[0]['antenna_weights']))
            self.params['resolution'] = first_sample_params.get('resolution', heatmap_shape[0])
        else:
            raise ValueError("Dataset is empty. Cannot infer parameters.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        heatmap = np.array(sample['heatmap'], dtype=np.float32)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)
        
        weights = sample['antenna_weights']
        weights_array = np.zeros(len(weights) * 2, dtype=np.float32)
        for i, w in enumerate(weights):
            weights_array[2 * i] = w['real']
            weights_array[2 * i + 1] = w['imag']
            
        weights_tensor = torch.from_numpy(weights_array)
        
        return heatmap_tensor, weights_tensor

def create_dummy_dataset(json_path: Path, num_antennas: int, resolution: int):
    """Creates a small random dataset with specified parameters."""
    dummy_data = []
    for _ in range(50):
        sample = {
            "heatmap": np.random.rand(resolution, resolution).tolist(),
            "antenna_weights": [{"real": np.random.randn(), "imag": np.random.randn()} for _ in range(num_antennas)],
            "coverage_percent": np.random.uniform(50, 90),
            "params": {"num_antennas": num_antennas, "resolution": resolution}
        }
        dummy_data.append(sample)
    with open(json_path, 'w') as f:
        json.dump(dummy_data, f)


# --- 2. Model Architecture: A Simple CNN for Regression ---

class BeamformingCNN(nn.Module):
    """CNN to predict antenna weights from a UE heatmap."""
    def __init__(self, num_antennas, resolution):
        super(BeamformingCNN, self).__init__()
        self.output_size = num_antennas * 2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        dummy_input = torch.randn(1, 1, resolution, resolution)
        flattened_size = self.encoder(dummy_input).reshape(-1).shape[0]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.regressor(x)
        return x

# --- 3. Physics Simulation (for Validation) ---

def calculate_array_factor_torch(theta_deg, weights_tensor):
    """Calculates array factor from weights tensor. Num antennas is inferred."""
    num_antennas = weights_tensor.shape[1] // 2
    weights_complex = torch.view_as_complex(weights_tensor.reshape(-1, num_antennas, 2))

    theta_rad = torch.deg2rad(theta_deg)  # shape (num_angles,)
    k_d = torch.pi  # Equivalent to 2 * pi * 0.5 (d_lambda)
    n = torch.arange(num_antennas, device=weights_tensor.device).float()  # shape (num_antennas,)

    # Reshape for broadcasting
    n = n[:, None]                     # (num_antennas, 1)
    theta_rad = theta_rad[None, :]     # (1, num_angles)

    steering_vector = torch.exp(-1j * n * k_d * torch.sin(theta_rad))  # (num_antennas, num_angles)

    # Batch multiply: (batch, num_antennas) × (num_antennas, num_angles)
    total_field = torch.matmul(weights_complex, steering_vector)  # (batch, num_angles)
    
    return torch.abs(total_field) / num_antennas


# --- 4. Main Training and Evaluation Script ---

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset to infer parameters.
    # The constructor uses CLI args as a fallback if the file needs to be created.
    dataset = MIMODataset(Path(args.dataset), args.num_antennas, args.resolution)
    
    # Determine the final configuration, prioritizing CLI args over inferred values.
    inferred_res = dataset.params['resolution']
    inferred_ant = dataset.params['num_antennas']
    
    RESOLUTION = args.resolution if args.resolution is not None else inferred_res
    NUM_ANTENNAS = args.num_antennas if args.num_antennas is not None else inferred_ant
    
    print("\n--- Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Antenna Elements: {NUM_ANTENNAS}")
    print("---------------------\n")

    if args.resolution and args.resolution != inferred_res:
        print(f"Warning: CLI --resolution ({args.resolution}) overrides dataset value ({inferred_res}).")
    if args.num_antennas and args.num_antennas != inferred_ant:
        print(f"Warning: CLI --num-antennas ({args.num_antennas}) overrides dataset value ({inferred_ant}).")

    train_size = int(0.7 * len(dataset)); val_size = int(0.15 * len(dataset)); test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test.")
    
    model = BeamformingCNN(num_antennas=NUM_ANTENNAS, resolution=RESOLUTION).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'val_loss': []}
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for heatmaps, weights in train_loader:
            heatmaps, weights = heatmaps.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(heatmaps)
            loss = criterion(outputs, weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history['train_loss'].append(train_loss / len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for heatmaps, weights in val_loader:
                heatmaps, weights = heatmaps.to(device), weights.to(device)
                outputs = model(heatmaps)
                val_loss += criterion(outputs, weights).item()
        history['val_loss'].append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {history['train_loss'][-1]:.6f} | Val Loss: {history['val_loss'][-1]:.6f}")
    print("--- Training Finished ---\n")
    
    # *** FIX: Pass the globally determined NUM_ANTENNAS to the evaluation function ***
    evaluate_and_plot(model, test_loader, history, device, num_antennas=NUM_ANTENNAS)


def evaluate_and_plot(model, test_loader, history, device, num_antennas):
    """Evaluate the model and generate publication-ready plots."""
    model.eval()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history['train_loss'], label='Training Loss', color='b')
    ax.plot(history['val_loss'], label='Validation Loss', color='r')
    ax.set_title('Model Training and Validation Loss', fontsize=20); ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Mean Squared Error (MSE)', fontsize=20)
    ax.legend(fontsize=12); ax.grid(True)
    fig.tight_layout()
    plt.savefig('training.png', dpi=300)
    print("Saved training loss curve to 'training.png'")
    plt.close(fig)

    all_predictions = []; all_labels = []
    with torch.no_grad():
        for heatmaps, labels in test_loader:
            outputs = model(heatmaps.to(device))
            all_predictions.append(outputs.cpu())
            all_labels.append(labels)
            
    predictions_tensor = torch.cat(all_predictions, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # *** FIX: The `num_antennas` parameter is now trusted, as it comes from the model's configuration ***
    pred_complex = torch.view_as_complex(predictions_tensor.reshape(-1, num_antennas, 2)).cpu().numpy()
    label_complex = torch.view_as_complex(labels_tensor.reshape(-1, num_antennas, 2)).cpu().numpy()

    pred_mag = np.abs(pred_complex).flatten(); label_mag = np.abs(label_complex).flatten()
    pred_phase = np.angle(pred_complex, deg=True).flatten(); label_phase = np.angle(label_complex, deg=True).flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.scatter(label_mag, pred_mag, alpha=0.3, edgecolors='none'); ax1.plot([min(label_mag), max(label_mag)], [min(label_mag), max(label_mag)], 'r--', lw=2, label='Ideal')
    ax1.set_title('Weight Magnitude Prediction', fontsize=20); ax1.set_xlabel('Actual Magnitude', fontsize=12); ax1.set_ylabel('Predicted Magnitude', fontsize=12)
    ax1.grid(True); ax1.legend()
    
    ax2.scatter(label_phase, pred_phase, alpha=0.3, edgecolors='none'); ax2.plot([-180, 180], [-180, 180], 'r--', lw=2, label='Ideal')
    ax2.set_title('Weight Phase Prediction', fontsize=20); ax2.set_xlabel('Actual Phase (°)', fontsize=12); ax2.set_ylabel('Predicted Phase (°)', fontsize=12)
    ax2.set_xticks([-180, -90, 0, 90, 180]); ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.grid(True); ax2.legend()

    fig.tight_layout()
    plt.savefig('evaluation.png', dpi=300)
    print("Saved prediction scatter plot to 'evaluation.png'")
    plt.close(fig)
    
    angles = torch.linspace(-90, 90, 361, device=device)
    actual_weights_sample = labels_tensor[0:1].to(device)
    pred_weights_sample = predictions_tensor[0:1].to(device)
    actual_pattern = calculate_array_factor_torch(angles, actual_weights_sample).cpu().numpy().flatten()
    pred_pattern = calculate_array_factor_torch(angles, pred_weights_sample).cpu().numpy().flatten()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(np.deg2rad(angles.cpu().numpy()), actual_pattern, label='Actual (Baseline)', lw=3)
    ax.plot(np.deg2rad(angles.cpu().numpy()), pred_pattern, label='Predicted (ML Model)', linestyle='--', lw=2)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_thetamin(-90); ax.set_thetamax(90)
    ax.set_rlabel_position(22.5); ax.set_title('Beam Pattern Comparison on a Test Sample', fontsize=20)
    ax.legend(); fig.tight_layout()
    plt.savefig('comparison.png', dpi=300)
    print("Saved beam pattern comparison to 'comparison.png'")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN for MIMO Beamforming Prediction.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--num-antennas', type=int, default=None, help="Manually override number of antenna elements from dataset.")
    parser.add_argument('--resolution', type=int, default=None, help="Manually override heatmap resolution from dataset.")
    
    args = parser.parse_args()
    main(args)