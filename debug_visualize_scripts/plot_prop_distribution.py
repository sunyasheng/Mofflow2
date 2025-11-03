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

print(f"\nStatistics:")
print(f"  Total entries: {len(prop_values)}")
print(f"  Mean: {np.mean(prop_values):.6f}")
print(f"  Median: {np.median(prop_values):.6f}")
print(f"  Std: {np.std(prop_values):.6f}")
print(f"  Min: {np.min(prop_values):.6f}")
print(f"  Max: {np.max(prop_values):.6f}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Property Values', fontsize=16, fontweight='bold')

# 1. Histogram
axes[0, 0].hist(prop_values, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Property Value', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 2. Density plot (KDE)
axes[0, 1].hist(prop_values, bins=50, density=True, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 1].set_xlabel('Property Value', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('Density Plot', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot
axes[1, 0].boxplot(prop_values, vert=True)
axes[1, 0].set_ylabel('Property Value', fontsize=12)
axes[1, 0].set_title('Box Plot', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative distribution
sorted_props = np.sort(prop_values)
cumulative = np.arange(1, len(sorted_props) + 1) / len(sorted_props)
axes[1, 1].plot(sorted_props, cumulative, linewidth=2)
axes[1, 1].set_xlabel('Property Value', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1, 1].set_title('Cumulative Distribution Function', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_path = Path(file_path).parent / 'prop_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Also show the plot
plt.show()

