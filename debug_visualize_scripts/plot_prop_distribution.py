#!/usr/bin/env python3
"""Script to plot the distribution of 'prop' values from mof_sequence_val.json"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Get file path from command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_prop_distribution.py <path_to_json_file>")
    print("Example: python plot_prop_distribution.py /ibex/project/c2318/material_discovery/MOFFLOW2_data/seqs/mof_sequence_val.json")
    sys.exit(1)

file_path = sys.argv[1]

if not Path(file_path).exists():
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

print(f"Reading from: {file_path}")
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract all prop values
prop_values = []
for key, entry in data.items():
    if 'prop' in entry:
        prop_values.append(entry['prop'])

if not prop_values:
    print("Error: No 'prop' values found in the data")
    sys.exit(1)

prop_values = np.array(prop_values)

# Calculate statistics
mean_val = np.mean(prop_values)
median_val = np.median(prop_values)
std_val = np.std(prop_values)
min_val = np.min(prop_values)
max_val = np.max(prop_values)
q25 = np.percentile(prop_values, 25)
q75 = np.percentile(prop_values, 75)
q95 = np.percentile(prop_values, 95)
q99 = np.percentile(prop_values, 99)

print(f"\nStatistics:")
print(f"  Total entries: {len(prop_values)}")
print(f"  Mean: {mean_val:.6f}")
print(f"  Median: {median_val:.6f}")
print(f"  Std: {std_val:.6f}")
print(f"  Min: {min_val:.6f}")
print(f"  Max: {max_val:.6f}")
print(f"  25th percentile: {q25:.6f}")
print(f"  75th percentile: {q75:.6f}")
print(f"  95th percentile: {q95:.6f}")
print(f"  99th percentile: {q99:.6f}")

# Determine if data is highly skewed (most values in a small range)
data_range = max_val - min_val
concentrated_range = q95 - min_val
skew_ratio = concentrated_range / data_range if data_range > 0 else 1.0

print(f"\nData distribution analysis:")
print(f"  Full range: {data_range:.6f}")
print(f"  95% of data within: {concentrated_range:.6f}")
print(f"  Skew ratio: {skew_ratio:.4f} (closer to 0 = more concentrated)")

# Determine optimal x-axis range
# Focus on range that contains most data
if skew_ratio < 0.2:  # Highly skewed
    xlim_main = (min_val, q95 * 1.1)  # Focus on main concentration
    xlim_full = (min_val, max_val)
    print(f"  Using focused range: [{xlim_main[0]:.6f}, {xlim_main[1]:.6f}]")
else:
    xlim_main = (min_val, max_val)
    xlim_full = xlim_main

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Property Values', fontsize=16, fontweight='bold')

# 1. Histogram with focused range
n_bins = min(100, int(len(prop_values) / 10))
axes[0, 0].hist(prop_values, bins=n_bins, edgecolor='black', alpha=0.7, range=xlim_main)
axes[0, 0].set_xlabel('Property Value', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title(f'Histogram (focused range: 0 to {xlim_main[1]:.4f})', fontsize=14)
axes[0, 0].set_xlim(xlim_main)
axes[0, 0].grid(True, alpha=0.3)

# 2. Density plot with focused range
axes[0, 1].hist(prop_values, bins=n_bins, density=True, edgecolor='black', alpha=0.7, color='skyblue', range=xlim_main)
axes[0, 1].set_xlabel('Property Value', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title(f'Density Plot (focused range)', fontsize=14)
axes[0, 1].set_xlim(xlim_main)
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot with focused y-axis
bp = axes[1, 0].boxplot(prop_values, vert=True, whis=(5, 95))
axes[1, 0].set_ylabel('Property Value', fontsize=12)
axes[1, 0].set_title('Box Plot (showing 5th-95th percentile)', fontsize=14)
axes[1, 0].set_ylim(xlim_main)
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative distribution with focused range
sorted_props = np.sort(prop_values)
cumulative = np.arange(1, len(sorted_props) + 1) / len(sorted_props)
axes[1, 1].plot(sorted_props, cumulative, linewidth=2)
axes[1, 1].set_xlabel('Property Value', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 1].set_title('Cumulative Distribution Function', fontsize=14)
axes[1, 1].set_xlim(xlim_main)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(q95, color='r', linestyle='--', alpha=0.5, label=f'95th percentile: {q95:.4f}')
axes[1, 1].legend()

plt.tight_layout()

# Save the focused plot
output_path = Path(file_path).parent / 'prop_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Create a second figure showing zoomed-in view of main concentration
if skew_ratio < 0.2:
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Distribution of Property Values (Zoomed View)', fontsize=16, fontweight='bold')
    
    # Even more focused range - show where most data is
    xlim_zoom = (min_val, q75 * 1.2)
    
    # 1. Histogram zoomed
    axes2[0, 0].hist(prop_values, bins=n_bins, edgecolor='black', alpha=0.7, range=xlim_zoom)
    axes2[0, 0].set_xlabel('Property Value', fontsize=12)
    axes2[0, 0].set_ylabel('Frequency', fontsize=12)
    axes2[0, 0].set_title(f'Histogram (zoomed: 0 to {xlim_zoom[1]:.4f})', fontsize=14)
    axes2[0, 0].set_xlim(xlim_zoom)
    axes2[0, 0].grid(True, alpha=0.3)
    
    # 2. Density plot zoomed
    axes2[0, 1].hist(prop_values, bins=n_bins, density=True, edgecolor='black', alpha=0.7, color='skyblue', range=xlim_zoom)
    axes2[0, 1].set_xlabel('Property Value', fontsize=12)
    axes2[0, 1].set_ylabel('Density', fontsize=12)
    axes2[0, 1].set_title('Density Plot (zoomed)', fontsize=14)
    axes2[0, 1].set_xlim(xlim_zoom)
    axes2[0, 1].grid(True, alpha=0.3)
    
    # 3. Log scale histogram (if values > 0)
    if min_val > 0:
        axes2[1, 0].hist(prop_values, bins=n_bins, edgecolor='black', alpha=0.7, color='green')
        axes2[1, 0].set_yscale('log')
        axes2[1, 0].set_xlabel('Property Value', fontsize=12)
        axes2[1, 0].set_ylabel('Frequency (log scale)', fontsize=12)
        axes2[1, 0].set_title('Histogram (log y-scale)', fontsize=14)
        axes2[1, 0].set_xlim(xlim_main)
        axes2[1, 0].grid(True, alpha=0.3)
    else:
        axes2[1, 0].text(0.5, 0.5, 'Log scale not applicable\n(min value = 0)', 
                        ha='center', va='center', transform=axes2[1, 0].transAxes, fontsize=12)
        axes2[1, 0].set_title('Log Scale (not applicable)', fontsize=14)
    
    # 4. Cumulative distribution zoomed
    axes2[1, 1].plot(sorted_props, cumulative, linewidth=2)
    axes2[1, 1].set_xlabel('Property Value', fontsize=12)
    axes2[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
    axes2[1, 1].set_title('CDF (zoomed)', fontsize=14)
    axes2[1, 1].set_xlim(xlim_zoom)
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].axvline(q75, color='r', linestyle='--', alpha=0.5, label=f'75th percentile: {q75:.4f}')
    axes2[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the zoomed plot
    output_path_zoom = Path(file_path).parent / 'prop_distribution_zoomed.png'
    fig2.savefig(output_path_zoom, dpi=300, bbox_inches='tight')
    print(f"Zoomed plot saved to: {output_path_zoom}")

# Show the plot(s)
plt.show()

