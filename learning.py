import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Configuration ---
DATASET_FILE = 'dataset.json' # Make sure to generate this file
RESOLUTION = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Create a dummy dataset file for demonstration if it doesn't exist ---
if not os.path.exists(DATASET_FILE):
    print(f"'{DATASET_FILE}' not found. Creating a small dummy dataset for demonstration.")
    dummy_data = []
    for _ in range(100):
        # Create a plausible-looking random heatmap
        heatmap = np.random.rand(RESOLUTION, RESOLUTION) * 0.1
        # Add a random hotspot
        cx, cy = np.random.randint(0, RESOLUTION, 2)
        radius = np.random.randint(10, 20)
        x, y = np.ogrid[:RESOLUTION, :RESOLUTION]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        heatmap[dist_from_center <= radius] += 0.9 * np.exp(-dist_from_center[dist_from_center <= radius]/5)
        heatmap /= np.max(heatmap)
        
        # Calculate a plausible-looking angle (simplified)
        antennaX, antennaY = RESOLUTION / 2, RESOLUTION
        dx = cx - antennaX
        dy = antennaY - cy
        angle = np.rad2deg(np.arctan2(dx, dy))
        
        dummy_data.append({'heatmap': heatmap.tolist(), 'steering_angle_degrees': angle})

    with open(DATASET_FILE, 'w') as f:
        json.dump(dummy_data, f)


# --- 1. Data Loading and Preprocessing ---

class HeatmapDataset(Dataset):
    """PyTorch Dataset for loading heatmap data from JSON."""
    def __init__(self, json_file):
        print(f"Loading data from {json_file}...")
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print("\n" + "="*50)
            print(f" ERROR: Dataset file '{json_file}' not found.")
            print(" Please generate it using the HTML simulator tool first.")
            print("="*50 + "\n")
            self.data = [{'heatmap': np.zeros((RESOLUTION, RESOLUTION)).tolist(), 'steering_angle_degrees': 0.0}]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        heatmap = torch.tensor(item['heatmap'], dtype=torch.float32).unsqueeze(0)
        angle = item['steering_angle_degrees'] / 90.0
        label = torch.tensor([angle], dtype=torch.float32)
        return heatmap, label

# --- 2. CNN Model Architecture ---

class BeamformingCNN(nn.Module):
    """A simple CNN to predict steering angle from a heatmap."""
    def __init__(self, input_size=128):
        super(BeamformingCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.flattened_size = 64 * (input_size // 16) * (input_size // 16)
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

# --- 3. Training Logic ---

def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print("--- Training Finished ---")
    return model, history

# --- 4. Evaluation and Visualization ---

def plot_training_history(history):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Training Loss', color='royalblue', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
    ax.set_title('Model Training and Validation Loss', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=300)
    print("\nSaved training loss curve to 'training_loss_curve.png'")
    plt.show()

def evaluate_and_visualize(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_preds.extend(outputs.numpy().flatten() * 90.0)
            all_labels.extend(labels.numpy().flatten() * 90.0)
            
    mse = np.mean((np.array(all_preds) - np.array(all_labels))**2)
    print(f"\n--- Evaluation on Test Set ---")
    print(f"Mean Squared Error (MSE) on angles: {mse:.4f} degrees^2")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f} degrees")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_labels, all_preds, alpha=0.6, edgecolors='k', color='royalblue', label='Predictions')
    ax.plot([-90, 90], [-90, 90], 'r--', linewidth=2, label='Ideal Performance')
    ax.set_title('Predicted vs. Actual Steering Angles', fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Angle (Degrees)', fontsize=12)
    ax.set_ylabel('Predicted Angle (Degrees)', fontsize=12)
    ax.set_xlim(-90, 90); ax.set_ylim(-90, 90)
    ax.set_aspect('equal', adjustable='box'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png', dpi=300)
    print("Saved prediction scatter plot to 'predicted_vs_actual.png'")
    plt.show()

    num_examples = 3
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    fig.suptitle('Visual Comparison of Beamforming Performance', fontsize=20, fontweight='bold')
    test_iter = iter(test_loader)
    for i in range(num_examples):
        heatmap_tensor, label_tensor = next(test_iter)
        heatmap = heatmap_tensor[0, 0].numpy()
        actual_angle = label_tensor[0].item() * 90.0
        pred_angle = model(heatmap_tensor[0].unsqueeze(0)).item() * 90.0

        axes[i, 0].imshow(heatmap, cmap='viridis')
        axes[i, 0].set_title(f'Input Heatmap (Sample {i+1})'); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].add_patch(patches.Circle((RESOLUTION/2, RESOLUTION), 5, color='red', fill=True))
        
        plot_beam_pattern_on_ax(axes[i, 1], actual_angle); axes[i, 1].set_title(f'Baseline Angle: {actual_angle:.2f}°')
        plot_beam_pattern_on_ax(axes[i, 2], pred_angle); axes[i, 2].set_title(f'ML Predicted Angle: {pred_angle:.2f}°')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('beam_comparison.png', dpi=300)
    print("Saved beam pattern comparison to 'beam_comparison.png'")
    plt.show()

def calculate_array_factor(theta_deg, theta0_deg, N=16, d_lambda=0.5):
    theta, theta0 = np.deg2rad(theta_deg), np.deg2rad(theta0_deg)
    psi = 2 * np.pi * d_lambda * (np.sin(theta) - np.sin(theta0))
    with np.errstate(divide='ignore', invalid='ignore'):
      af = np.sin(N * psi / 2) / (N * np.sin(psi / 2))
    return np.abs(np.nan_to_num(af, nan=1.0))

def plot_beam_pattern_on_ax(ax, steering_angle):
    angles_rad = np.deg2rad(np.arange(-180, 181, 1))
    gains = calculate_array_factor(np.arange(-180, 181, 1), steering_angle)
    ax.clear()
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.plot(angles_rad, gains, color='#d35400', linewidth=2)
    ax.fill(angles_rad, gains, alpha=0.2, color='#d35400')
    ax.set_ylim(0, 1.1); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(['-3dB', '0dB'])
    ax.set_thetagrids(range(0, 360, 45)); ax.set_rlabel_position(22.5)

# --- Main Execution ---
if __name__ == "__main__":
    full_dataset = HeatmapDataset(DATASET_FILE)
    train_size = int(TRAIN_TEST_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = BeamformingCNN(input_size=RESOLUTION)
    model, history = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    plot_training_history(history)
    evaluate_and_visualize(model, test_loader)