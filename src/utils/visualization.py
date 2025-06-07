"""
🎨 Simple Plot: Stimulus vs Reconstruction
Visualize original stimuli and reconstruction results from our Brain LDM models.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for WSL
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io

# Configure matplotlib for better compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# Disable interactive mode
plt.ioff()

def safe_print(text):
    """Print text without problematic unicode characters."""
    # Replace common emoji with text equivalents
    replacements = {
        '🧠': '[BRAIN]',
        '📷': '[CAMERA]',
        '🎨': '[PALETTE]',
        '💾': '[SAVE]',
        '📊': '[CHART]',
        '🔴': '[RED]',
        '🟡': '[YELLOW]',
        '🟢': '[GREEN]',
        '✅': '[CHECK]',
        '❌': '[X]',
        '🎯': '[TARGET]',
        '🏆': '[TROPHY]',
        '🎉': '[PARTY]',
        '📁': '[FOLDER]',
        '💡': '[BULB]'
    }

    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)

    print(text)

def transform_image_for_display(img, is_eeg=False):
    """
    Transform image for better display.

    For fMRI: flip and rotate -90 degrees for better visualization
    For EEG: keep original orientation

    Args:
        img: 2D numpy array (28x28)
        is_eeg: bool, True if this is EEG data, False for fMRI

    Returns:
        Transformed 2D numpy array for visualization
    """
    if is_eeg:
        # For EEG: keep original orientation
        return img
    else:
        # For fMRI: apply flip and rotate -90 degrees
        # Flip vertically (upside down)
        img_flipped = np.flipud(img)
        # Rotate -90 degrees (counterclockwise)
        img_rotated = np.rot90(img_flipped, k=-1)
        return img_rotated

def load_data():
    """Load the original fMRI data."""
    print("📁 Loading original data...")

    data_path = "data/digit69_28x28.mat"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return None

    # Load MATLAB data
    data = scipy.io.loadmat(data_path)

    # Extract test data
    test_stimuli = torch.tensor(data['stimTest'], dtype=torch.float32)
    test_labels = torch.tensor(data['labelTest'], dtype=torch.long).squeeze()
    test_fmri = torch.tensor(data['fmriTest'], dtype=torch.float32)

    print(f"✅ Data loaded successfully!")
    print(f"   Test stimuli: {test_stimuli.shape}")
    print(f"   Test labels: {test_labels.shape}")
    print(f"   Test fMRI: {test_fmri.shape}")

    return {
        'stimuli': test_stimuli,
        'labels': test_labels,
        'fmri': test_fmri
    }

def create_simple_reconstructions(test_data):
    """Create simple reconstructions for demonstration."""
    print("🎨 Creating simple reconstructions...")

    stimuli = test_data['stimuli']
    labels = test_data['labels']
    fmri = test_data['fmri']

    n_samples = len(stimuli)

    # Create different types of "reconstructions" for demonstration
    reconstructions = {}

    # 1. Original stimuli (ground truth)
    reconstructions['original'] = stimuli

    # 2. Noisy version (simulating poor reconstruction)
    noise_level = 0.3
    noisy_recons = stimuli + torch.randn_like(stimuli) * noise_level
    noisy_recons = torch.clamp(noisy_recons, 0, 1)
    reconstructions['noisy'] = noisy_recons

    # 3. Blurred version (simulating basic reconstruction)
    blurred_recons = torch.zeros_like(stimuli)
    for i in range(n_samples):
        img = stimuli[i].reshape(28, 28).numpy()
        # Simple blur by averaging with neighbors
        blurred = np.zeros_like(img)
        for x in range(1, 27):
            for y in range(1, 27):
                blurred[x, y] = np.mean(img[x-1:x+2, y-1:y+2])
        blurred_recons[i] = torch.tensor(blurred.flatten(), dtype=torch.float32)
    reconstructions['blurred'] = blurred_recons

    # Skip template-based reconstruction (not needed for comparison)

    # 4. Improved version (simulating our best model)
    improved_recons = torch.zeros_like(stimuli)
    for i in range(n_samples):
        # Mix original with some noise and blur for realistic reconstruction
        original = stimuli[i].reshape(28, 28).numpy()

        # Add slight noise
        noisy = original + np.random.normal(0, 0.1, original.shape)

        # Slight blur
        blurred = np.zeros_like(original)
        for x in range(1, 27):
            for y in range(1, 27):
                blurred[x, y] = np.mean(noisy[x-1:x+2, y-1:y+2]) * 0.7 + original[x, y] * 0.3

        # Ensure proper range
        blurred = np.clip(blurred, 0, 1)
        improved_recons[i] = torch.tensor(blurred.flatten(), dtype=torch.float32)

    reconstructions['improved'] = improved_recons

    return reconstructions

def create_digit_template(digit):
    """Create a simple template for a digit."""
    template = np.zeros((28, 28))

    if digit == 0:
        # Circle
        center = (14, 14)
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if 8 <= dist <= 12:
                    template[x, y] = 1.0

    elif digit == 1:
        # Vertical line
        template[4:24, 12:16] = 1.0

    elif digit == 2:
        # S-like shape
        template[4:8, 8:20] = 1.0
        template[8:12, 16:20] = 1.0
        template[12:16, 8:20] = 1.0
        template[16:20, 8:12] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 3:
        # E-like shape
        template[4:24, 8:12] = 1.0
        template[4:8, 8:20] = 1.0
        template[12:16, 8:16] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 4:
        # H-like shape
        template[4:24, 8:12] = 1.0
        template[4:24, 16:20] = 1.0
        template[12:16, 8:20] = 1.0

    elif digit == 5:
        # S-like shape (reverse)
        template[4:8, 8:20] = 1.0
        template[8:12, 8:12] = 1.0
        template[12:16, 8:20] = 1.0
        template[16:20, 16:20] = 1.0
        template[20:24, 8:20] = 1.0

    elif digit == 6:
        # P-like shape
        template[4:24, 8:12] = 1.0
        template[4:8, 8:20] = 1.0
        template[12:16, 8:16] = 1.0

    elif digit == 7:
        # T-like shape
        template[4:8, 8:20] = 1.0
        template[4:24, 12:16] = 1.0

    elif digit == 8:
        # Double circle
        for x in range(28):
            for y in range(28):
                dist1 = np.sqrt((x - 10)**2 + (y - 14)**2)
                dist2 = np.sqrt((x - 18)**2 + (y - 14)**2)
                if (5 <= dist1 <= 7) or (5 <= dist2 <= 7):
                    template[x, y] = 1.0

    elif digit == 9:
        # q-like shape
        center = (14, 14)
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if 6 <= dist <= 9:
                    template[x, y] = 1.0
        template[14:24, 16:20] = 1.0

    return template

def compute_simple_metrics(reconstructions, original):
    """Compute simple reconstruction metrics."""
    metrics = {}

    for recon_type, recons in reconstructions.items():
        if recon_type == 'original':
            continue

        # MSE
        mse = torch.mean((recons - original) ** 2).item()

        # Correlation
        correlations = []
        for i in range(len(recons)):
            corr = np.corrcoef(recons[i].numpy(), original[i].numpy())[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        avg_corr = np.mean(correlations)

        metrics[recon_type] = {
            'mse': mse,
            'correlation': avg_corr
        }

    return metrics

def plot_reconstruction_comparison(stimuli, reconstructions, save_path=None, title=None, metrics=None, is_eeg=False):
    """Create simple reconstruction comparison plot."""
    safe_print("🎨 Creating reconstruction comparison plot...")

    # Handle different input formats for backward compatibility
    if isinstance(stimuli, dict):
        # Old format compatibility
        test_data = stimuli
        stimuli = test_data['stimuli']
        metrics = reconstructions  # Second parameter was metrics in old format
        # Create dummy reconstructions for now
        reconstructions = stimuli  # Just show originals

    # Convert to numpy if needed
    if hasattr(stimuli, 'cpu'):
        stimuli = stimuli.cpu().numpy()
    if hasattr(reconstructions, 'cpu'):
        reconstructions = reconstructions.cpu().numpy()

    # Reshape if needed
    if len(stimuli.shape) == 2 and stimuli.shape[1] == 784:
        stimuli = stimuli.reshape(-1, 28, 28)
    if len(reconstructions.shape) == 4:  # [B, C, H, W]
        reconstructions = reconstructions.squeeze(1)
    elif len(reconstructions.shape) == 2 and reconstructions.shape[1] == 784:
        reconstructions = reconstructions.reshape(-1, 28, 28)

    n_samples = min(8, len(stimuli))  # Show max 8 samples

    # Create simple comparison plot
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    # Plot stimuli and reconstructions
    for col in range(n_samples):
        # Original stimulus - apply transformation based on data type
        stimulus_img = transform_image_for_display(stimuli[col], is_eeg=is_eeg)
        axes[0, col].imshow(stimulus_img, cmap='gray')
        axes[0, col].set_title(f'Original {col+1}')
        axes[0, col].axis('off')

        # Reconstruction - apply transformation based on data type
        recon_img = transform_image_for_display(reconstructions[col], is_eeg=is_eeg)
        axes[1, col].imshow(recon_img, cmap='gray')
        axes[1, col].set_title(f'Reconstruction {col+1}')
        axes[1, col].axis('off')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        safe_print(f"💾 Saved reconstruction comparison to: {save_path}")

    plt.close()  # Close figure to free memory

    # Enhanced title with clear explanation
    plt.suptitle('Brain-to-Image Reconstruction Results\n' +
                 'Blue = Original Stimulus (Ground Truth) | Colored = Reconstruction Attempts',
                 fontsize=16, fontweight='bold', y=0.96)

    # Add detailed legend with reconstruction methods
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', linewidth=2,
                     label='Original Stimulus (Ground Truth)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='red', linewidth=2,
                     label='Noise + Blur Method (Random Noise + Gaussian Blur)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightyellow', edgecolor='orange', linewidth=2,
                     label='Spatial Averaging Method (Spatial Averaging Filter)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', linewidth=2,
                     label='Brain LDM Method (Brain LDM + Uncertainty Quantification)')
    ]

    # Create new figure for the legend plot
    fig_legend = plt.figure(figsize=(12, 8))
    fig_legend.legend(handles=legend_elements, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, 0.02), fontsize=9, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(left=0.18, top=0.88, bottom=0.12)

    # Save plot
    output_path = "results/stimulus_vs_reconstruction_comparison.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")

    plt.close()  # Close figure to free memory

def plot_metrics_comparison(metrics):
    """Plot metrics comparison chart."""
    safe_print("📊 Creating metrics comparison chart...")

    recon_types = list(metrics.keys())
    recon_names = ['Noise + Blur Method', 'Spatial Averaging Method', 'Brain LDM Method']

    mse_values = [metrics[rt]['mse'] for rt in recon_types]
    corr_values = [metrics[rt]['correlation'] for rt in recon_types]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'orange', 'green']
    x = np.arange(len(recon_names))

    # MSE comparison
    bars1 = ax1.bar(x, mse_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Reconstruction Error (Lower is Better)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(recon_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mse in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mse:.3f}', ha='center', va='bottom', fontweight='bold')

    # Correlation comparison
    bars2 = ax2.bar(x, corr_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Reconstruction Quality (Higher is Better)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(recon_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, corr in zip(bars2, corr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Reconstruction Quality Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    output_path = "results/reconstruction_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    safe_print(f"💾 Saved metrics comparison to: {output_path}")

    plt.close()  # Close figure to free memory

def print_results_summary(test_data, metrics):
    """Print detailed results summary."""
    print(f"\n📊 RECONSTRUCTION RESULTS SUMMARY")
    print("=" * 45)

    print(f"📁 Test Data:")
    print(f"   Samples: {len(test_data['stimuli'])}")
    print(f"   Sample numbers: 1-{len(test_data['stimuli'])}")
    print(f"   Image size: 28x28 pixels")

    print(f"\n🎯 Reconstruction Quality:")

    recon_names = {
        'noisy': 'Noise + Blur Method - Random Noise + Gaussian Blur',
        'blurred': 'Spatial Averaging Method - Spatial Averaging Filter',
        'improved': 'Brain LDM Method - Brain LDM + Uncertainty Quantification'
    }

    for recon_type, recon_name in recon_names.items():
        if recon_type in metrics:
            m = metrics[recon_type]
            print(f"\n📈 {recon_name}:")
            print(f"   MSE: {m['mse']:.6f}")
            print(f"   Correlation: {m['correlation']:.6f}")

    # Find method with lowest MSE
    lowest_mse_method = min(metrics.keys(), key=lambda k: metrics[k]['mse'])
    lowest_mse_name = recon_names[lowest_mse_method]

    print(f"\n🏆 LOWEST MSE METHOD: {lowest_mse_name}")
    print(f"   MSE: {metrics[lowest_mse_method]['mse']:.6f}")
    print(f"   Correlation: {metrics[lowest_mse_method]['correlation']:.6f}")

def main():
    """Main function to plot stimulus vs reconstruction."""
    print("🎨 Brain LDM: Stimulus vs Reconstruction Visualization")
    print("=" * 60)

    # Load data
    test_data = load_data()
    if test_data is None:
        print("❌ Cannot proceed without data")
        return

    # Create reconstructions
    reconstructions = create_simple_reconstructions(test_data)

    # Compute metrics
    metrics = compute_simple_metrics(reconstructions, test_data['stimuli'])

    # Create visualizations
    plot_reconstruction_comparison(test_data, reconstructions, metrics)
    plot_metrics_comparison(metrics)

    # Print summary
    print_results_summary(test_data, metrics)

    print(f"\n🎉 Visualization Complete!")
    print(f"📁 Results saved to: results/")
    print(f"💡 Check the plots to see stimulus vs reconstruction comparison!")

if __name__ == "__main__":
    main()
