import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse
import h5py 

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

def generate_heatmap(heatmap_type='random', num_blobs=10, min_sigma=10, max_sigma=20, anisotropic=False, correlated=False, occlusions=False, interference=False):
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    D = np.zeros((grid_size, grid_size))
    
    if heatmap_type == 'curved':
        # Curved path with blobs (enhanced with random curve)
        t = np.linspace(0, 1, num_blobs)
        cx = grid_size / 2 + (grid_size / 3) * np.sin(2 * np.pi * t)
        cy = grid_size * t
        blob_centers = list(zip(cx, cy))
    elif heatmap_type == 'random':
        # Random blob positions
        blob_centers = [(np.random.randint(20, grid_size-20), np.random.randint(20, grid_size-20)) for _ in range(num_blobs)]
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
        sigma = np.random.uniform(min_sigma, max_sigma)
        
        if anisotropic:
            sigma_x = np.random.uniform(min_sigma, max_sigma)
            sigma_y = np.random.uniform(min_sigma, max_sigma)
            rotation = np.random.uniform(0, 2*np.pi)
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            a = cos_r**2 / (2 * sigma_x**2) + sin_r**2 / (2 * sigma_y**2)
            b = -sin_r * cos_r / (2 * sigma_x**2) + sin_r * cos_r / (2 * sigma_y**2)
            c = sin_r**2 / (2 * sigma_x**2) + cos_r**2 / (2 * sigma_y**2)
            D += np.exp(-(a * (X - cx)**2 + 2 * b * (X - cx) * (Y - cy) + c * (Y - cy)**2))
        else:
            D += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    
    if correlated:
        # Add spatial autocorrelation via Gaussian filtering
        D = gaussian_filter(D, sigma=np.random.uniform(1, 3))
    
    if occlusions:
        # Add 1-3 random rectangular occlusions
        num_occl = np.random.randint(1, 4)
        for _ in range(num_occl):
            xmin = np.random.randint(0, grid_size//2)
            ymin = np.random.randint(0, grid_size//2)
            width = np.random.randint(10, 30)
            height = np.random.randint(10, 30)
            D[ymin:ymin+height, xmin:xmin+width] = 0
    
    if interference:
        # Simulate multi-cell interference: add a shifted secondary heatmap
        D_sec, _ = generate_heatmap(heatmap_type='random', num_blobs=num_blobs//2, min_sigma=min_sigma, max_sigma=max_sigma)
        D_sec = np.roll(D_sec, shift=grid_size//2, axis=1)  # Shift horizontally
        D += 0.3 * D_sec  # Add with reduced intensity
    
    D /= np.sum(D) if np.sum(D) > 0 else 1  # Normalize to probability density
    return D, blob_centers

# Steering vector functions
def a_h(th, N_h):
    return np.exp(1j * k0 * d * np.sin(th) * np.arange(N_h))

def a_v(ph, N_v):
    return np.exp(1j * k0 * d * np.cos(ph) * np.arange(N_v))

def compute_beams(D, grid_size, N_h, N_v):
    # Compute covariance matrix R
    M = N_h * N_v
    R = np.zeros((M, M), dtype=complex)
    theta = np.deg2rad((np.arange(grid_size) - grid_size/2) / (grid_size/2) * 60)
    phi = np.deg2rad(np.arange(grid_size) / grid_size * 60)
    THETA, PHI = np.meshgrid(theta, phi)
    for iy in range(grid_size):
        for ix in range(grid_size):
            th = THETA[iy, ix]
            ph = PHI[iy, ix]
            a = np.kron(a_v(ph, N_v), a_h(th, N_h))
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
                a = np.kron(a_v(ph, N_v), a_h(th, N_h))
                beam_patterns[k, iy, ix] = np.abs(np.dot(w.conj(), a))**2

    # Summed coverage
    total_coverage = np.sum(beam_patterns, axis=0)
    
    return eigenvalues, eigenvectors, beam_patterns, total_coverage, K

def visualize(D, blob_centers, eigenvalues, beam_patterns, total_coverage, K, heatmap_type):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Demand heatmap
    axs[0, 0].imshow(D, cmap='viridis', origin='lower')
    axs[0, 0].set_title(f'UE Demand Heatmap ({heatmap_type})')
    axs[0, 0].scatter(64, 0, marker='^', color='red', s=200, label='BS')  # BS marker
    if heatmap_type != 'uniform':
        for i in range(len(blob_centers) - 1):
            cx1, cy1 = blob_centers[i]
            cx2, cy2 = blob_centers[i+1]
            axs[0, 0].plot([cx1, cx2], [cy1, cy2], color='orange', lw=2)
    axs[0, 0].legend()

    # Eigenvalue scree plot
    axs[0, 1].plot(eigenvalues[:20], 'o-')  # Show top 20
    axs[0, 1].set_title('Top Eigenvalues (x:Mode, y:Value)')
    #axs[0, 1].set_xlabel('Eigenmode')
    #axs[0, 1].set_ylabel('Eigenvalue')

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

def generate_dataset(num_samples, heatmap_type, min_blobs, max_blobs, min_sigma, max_sigma, grid_size=128, N_h=16, N_v=16, output_file='dataset.h5'):
    Ds = []
    ws = []  # Optimal weights w* (top eigenvector)
    coverages = []  # Baseline coverage (top eigenvalue)
    params_list = []  # Store parameters for each sample
    
    for _ in range(num_samples):
        num_blobs_sample = np.random.randint(min_blobs, max_blobs + 1)
        anisotropic = np.random.choice([True, False])
        correlated = np.random.choice([True, False])
        occlusions = np.random.choice([True, False])
        interference = np.random.choice([True, False])
        D, _ = generate_heatmap(heatmap_type, num_blobs_sample, min_sigma, max_sigma, anisotropic, correlated, occlusions, interference)
        eigenvalues, eigenvectors, _, _, _ = compute_beams(D, grid_size, N_h, N_v)
        w_star = eigenvectors[:, 0]  # Top eigenvector
        baseline_coverage = eigenvalues[0] / np.sum(eigenvalues) * 100  # As percentage
        
        Ds.append(D)
        ws.append(w_star)
        coverages.append(baseline_coverage)
        params_list.append({
            'num_blobs': num_blobs_sample,
            'anisotropic': anisotropic,
            'correlated': correlated,
            'occlusions': occlusions,
            'interference': interference
        })
    
    # Global simulation parameters
    global_params = {
        'grid_size': grid_size,
        'N_h': N_h,
        'N_v': N_v,
        'N': N_h * N_v,
        'theta_deg_range': theta_deg_range,
        'phi_deg_range': phi_deg_range,
        'lambda': lam,
        'd': d
    }
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('Ds', data=np.array(Ds))
        f.create_dataset('ws', data=np.array(ws))
        f.create_dataset('coverages', data=np.array(coverages))
        # Save per-sample params as a compound dataset or separately
        dt = h5py.special_dtype(vlen=str)
        param_grp = f.create_group('sample_params')
        for i, p in enumerate(params_list):
            grp = param_grp.create_group(str(i))
            for key, val in p.items():
                grp.attrs[key] = str(val) if isinstance(val, bool) else val
        # Global params
        global_grp = f.create_group('global_params')
        for key, val in global_params.items():
            global_grp.attrs[key] = val
    
    print(f"Dataset generated and saved to {output_file} with {num_samples} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps, compute beam patterns, or generate datasets.")
    parser.add_argument('--heatmap_type', type=str, default='random', choices=['curved', 'random', 'uniform', 'linear', 'circular'], help="Type of heatmap to generate.")
    parser.add_argument('--num_blobs', type=int, default=10, help="Number of blobs for non-uniform heatmaps.")
    parser.add_argument('--min_sigma', type=int, default=10, help="Min sigma for Gaussian blobs.")
    parser.add_argument('--max_sigma', type=int, default=20, help="Max sigma for Gaussian blobs.")
    parser.add_argument('--generate', action='store_true', help="Flag to generate dataset instead of single run.")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples for dataset generation.")
    parser.add_argument('--output_file', type=str, default='dataset_1.h5', help="Output file for dataset (HDF5).")
    parser.add_argument('--min_blobs', type=int, default=1, help="Min number of blobs for random variation in dataset.")
    parser.add_argument('--max_blobs', type=int, default=10, help="Max number of blobs for random variation in dataset.")
    parser.add_argument('--grid_size', type=int, default=128, help="Grid size for heatmap.")
    parser.add_argument('--N_h', type=int, default=16, help="Horizontal antenna elements.")
    parser.add_argument('--N_v', type=int, default=16, help="Vertical antenna elements.")
    
    args = parser.parse_args()
    grid_size = args.grid_size
    N_h = args.N_h
    N_v = args.N_v
    N = N_h * N_v
    theta_deg_range = 60
    phi_deg_range = 60
    theta = np.deg2rad((np.arange(grid_size) - grid_size/2) / (grid_size/2) * theta_deg_range)
    phi = np.deg2rad(np.arange(grid_size) / grid_size * phi_deg_range)
    THETA, PHI = np.meshgrid(theta, phi)
    
    if args.generate:
        generate_dataset(args.num_samples, args.heatmap_type, args.min_blobs, args.max_blobs,
            args.min_sigma, args.max_sigma, args.grid_size, args.N_h, args.N_v, args.output_file)
    else:
        D, blob_centers = generate_heatmap(args.heatmap_type, args.num_blobs, args.min_sigma, args.max_sigma)
        eigenvalues, _, beam_patterns, total_coverage, K = compute_beams(D, grid_size, N_h, N_v)
        visualize(D, blob_centers, eigenvalues, beam_patterns, total_coverage, K, args.heatmap_type)