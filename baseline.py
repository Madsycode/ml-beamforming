import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py  # Added for better dataset storage with complex numbers

# Parameters
grid_size = 128
N_h = 16  # Horizontal elements
N_v = 16  # Vertical elements
lam = 1.0  # Wavelength
d = lam / 2.0  # Spacing
k0 = 2 * np.pi / lam
N = N_h * N_v  # Total antennas: 256

# Angular mapping (in radians)
theta_deg_range = 60  # +/- 60 deg for azimuth
phi_deg_range = 60    # 0 to 60 deg for elevation
theta = np.deg2rad((np.arange(grid_size) - grid_size/2) / (grid_size/2) * theta_deg_range)
phi = np.deg2rad(np.arange(grid_size) / grid_size * phi_deg_range)

THETA, PHI = np.meshgrid(theta, phi)  # THETA: horizontal, PHI: vertical

def generate_heatmap(heatmap_type='random', num_blobs=10, sigma=13):
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    D = np.zeros((grid_size, grid_size))
    
    if heatmap_type == 'curved':
        # Curved path with blobs
        blob_centers = [(64, 20), (55, 50), (45, 80), (60, 100), (80, 110)][:num_blobs]
    elif heatmap_type == 'random':
        # Random blob positions
        blob_centers = [(np.random.randint(20, 108), np.random.randint(20, 108)) for _ in range(num_blobs)]
    elif heatmap_type == 'uniform':
        # Uniform distribution
        D = np.ones((grid_size, grid_size))
        D /= np.sum(D)
        return D, []
    elif heatmap_type == 'linear':
        # Linear path from bottom to top
        blob_centers = [(64, i * grid_size // (num_blobs + 1)) for i in range(1, num_blobs + 1)]
    elif heatmap_type == 'circular':
        # Circular arrangement
        radius = grid_size / 3
        angles = np.linspace(0, 2 * np.pi, num_blobs, endpoint=False)
        blob_centers = [(64 + radius * np.cos(a), 64 + radius * np.sin(a)) for a in angles]
    else:
        raise ValueError("Unknown heatmap_type. Choose from: 'curved', 'random', 'uniform', 'linear', 'circular'")
    
    for cx, cy in blob_centers:
        cx, cy = min(max(cx, 0), grid_size-1), min(max(cy, 0), grid_size-1)
        D += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    
    D /= np.sum(D)  # Normalize to probability density
    return D, blob_centers

# Steering vector functions
def a_h(th):
    return np.exp(1j * k0 * d * np.sin(th) * np.arange(N_h))

def a_v(ph):
    return np.exp(1j * k0 * d * np.cos(ph) * np.arange(N_v))

def compute_beams(D):
    # Compute covariance matrix R
    M = N_h * N_v  # 256
    R = np.zeros((M, M), dtype=complex)
    for iy in range(grid_size):
        for ix in range(grid_size):
            th = THETA[iy, ix]
            ph = PHI[iy, ix]
            a = np.kron(a_v(ph), a_h(th))
            R += D[iy, ix] * np.outer(a, a.conj())

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top K beams (e.g., where cumulative > 95%)
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    K = np.argmax(cumulative >= 0.95) + 1
    print(f"Selected {K} beams to capture 95% of demand variance.")

    # Compute beam patterns on the grid
    beam_patterns = np.zeros((K, grid_size, grid_size))
    for k in range(K):
        w = eigenvectors[:, k]  # Beamforming vector
        for iy in range(grid_size):
            for ix in range(grid_size):
                th = THETA[iy, ix]
                ph = PHI[iy, ix]
                a = np.kron(a_v(ph), a_h(th))
                beam_patterns[k, iy, ix] = np.abs(np.dot(w.conj(), a))**2

    # Summed coverage
    total_coverage = np.sum(beam_patterns, axis=0)
    
    return eigenvalues, eigenvectors, beam_patterns, total_coverage, K

def visualize(D, blob_centers, eigenvalues, beam_patterns, total_coverage, K, heatmap_type):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Demand heatmap
    axs[0, 0].imshow(D, cmap='hot', origin='lower')
    axs[0, 0].set_title(f'UE Demand Heatmap ({heatmap_type})')
    axs[0, 0].scatter(64, 0, marker='^', color='blue', s=100, label='BS')  # BS marker
    if heatmap_type != 'uniform':
        for i in range(len(blob_centers) - 1):
            cx1, cy1 = blob_centers[i]
            cx2, cy2 = blob_centers[i+1]
            axs[0, 0].plot([cx1, cx2], [cy1, cy2], color='orange', lw=2)
    axs[0, 0].legend()

    # Eigenvalue scree plot
    axs[0, 1].plot(eigenvalues[:20], 'o-')  # Show top 20
    axs[0, 1].set_title('Top Eigenvalues')
    axs[0, 1].set_xlabel('Eigenmode')
    axs[0, 1].set_ylabel('Eigenvalue')

    # Summed beam coverage
    axs[0, 2].imshow(total_coverage, cmap='hot', origin='lower')
    axs[0, 2].set_title('Summed Beam Coverage')

    # Individual top beams (show first 3, or adjust if K < 3)
    num_to_show = min(K, 3)
    for i in range(num_to_show):
        row, col = 1, i
        axs[row, col].imshow(beam_patterns[i], cmap='hot', origin='lower')
        axs[row, col].set_title(f'Beam {i+1} Pattern')

    if K > 3:
        print(f"Note: Only showing first 3 beam patterns. Total K={K}. Modify code to show more if needed.")

    plt.tight_layout()
    plt.show()

def generate_dataset(num_samples, heatmap_type, min_blobs, max_blobs, min_sigma, max_sigma, output_file):
    Ds = []
    ws = []  # Optimal weights w* (top eigenvector)
    coverages = []  # Baseline coverage (top eigenvalue)
    
    for _ in range(num_samples):
        num_blobs_sample = np.random.randint(min_blobs, max_blobs + 1)
        sigma_sample = np.random.randint(min_sigma, max_sigma + 1)
        D, _ = generate_heatmap(heatmap_type, num_blobs_sample, sigma_sample)
        eigenvalues, eigenvectors, _, _, _ = compute_beams(D)
        w_star = eigenvectors[:, 0]  # Top eigenvector
        baseline_coverage = eigenvalues[0]  # Top eigenvalue as baseline coverage
        
        Ds.append(D)
        ws.append(w_star)
        coverages.append(baseline_coverage)
    
    # Simulation parameters
    params = {
        'S': grid_size,
        'N': N,
        'theta_deg_range': theta_deg_range,
        'phi_deg_range': phi_deg_range,
        'lambda': lam,
        'd': d
    }
    
    # Save to HDF5 for better handling of complex arrays and dict
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('Ds', data=np.array(Ds))
        f.create_dataset('ws', data=np.array(ws))
        f.create_dataset('coverages', data=np.array(coverages))
        # Save params as attributes
        grp = f.create_group('params')
        for key, val in params.items():
            grp.attrs[key] = val
    
    print(f"Dataset generated and saved to {output_file} with {num_samples} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps, compute beam patterns, or generate datasets.")
    parser.add_argument('--heatmap_type', type=str, default='random', choices=['curved', 'random', 'uniform', 'linear', 'circular'], help="Type of heatmap to generate.")
    
    parser.add_argument('--num_blobs', type=int, default=10, help="Number of blobs for non-uniform heatmaps.")
    parser.add_argument('--sigma', type=int, default=5, help="Sigma for Gaussian blobs.")
    parser.add_argument('--generate', action='store_true', help="Flag to generate dataset instead of single run.")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples for dataset generation.")
    parser.add_argument('--output_file', type=str, default='dataset.h5', help="Output file for dataset (HDF5).")
    parser.add_argument('--min_blobs', type=int, default=1, help="Min number of blobs for random variation in dataset.")
    parser.add_argument('--max_blobs', type=int, default=10, help="Max number of blobs for random variation in dataset.")
    parser.add_argument('--min_sigma', type=int, default=10, help="Min sigma for random variation in dataset.")
    parser.add_argument('--max_sigma', type=int, default=20, help="Max sigma for random variation in dataset.")
    
    args = parser.parse_args()
    
    if args.generate:
        # For dataset, force 'random' type for diversity, or use specified
        generate_dataset(args.num_samples, args.heatmap_type, args.min_blobs, args.max_blobs,
        args.min_sigma, args.max_sigma, args.output_file)
    else:
        D, blob_centers = generate_heatmap(args.heatmap_type, args.num_blobs, args.sigma)
        eigenvalues, _, beam_patterns, total_coverage, K = compute_beams(D)
        visualize(D, blob_centers, eigenvalues, beam_patterns, total_coverage, K, args.heatmap_type)