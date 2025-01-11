import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from scipy.interpolate import make_interp_spline

# Load and parse PMBW stats file
file_path = 'results-nps1/pmbw-stats.txt'

data = []
with open(file_path, 'r') as f:
    for line in f:
        if line.startswith("RESULT"):
            try:
                parts = line.split()
                record = {k: v for k, v in (pair.split('=') for pair in parts[1:] if '=' in pair)}
                data.append({
                    'areasize': int(record['areasize']),
                    'bandwidth': float(record['bandwidth']) / (1024**3),  # Convert to GiB/s
                    'nthreads': int(record['nthreads']),
                })
            except (ValueError, KeyError):
                continue

# Convert to DataFrame
df = pd.DataFrame(data)
df['log2_areasize'] = np.log2(df['areasize'])

# Define the new color palette
thread_counts = sorted(df['nthreads'].unique())
# Yellow to blue-green spectral - high contrast for data visualization
colors = [
    '#FFF7BC',  # pale yellow
    '#FEE391',  # warm yellow
    '#FEC44F',  # golden yellow
    '#85C4C9',  # light teal
    '#40C0A2',  # medium teal
    '#23BB99',  # deep teal
    '#096980',  # blue-green
    '#033F91'   # dark blue-green
]

'''
  const colors = {
    p48: '#142459',  // Darkest blue
    p32: '#176BA0',
    p24: '#19A4E1',
    p16: '#1AC6E3',
    p8: '#1BE4E6',
    p4: '#40F3E3',
    p2: '#7CFCE2',
    p1: '#C7FEE4'   // Lightest blue/mint
  };
'''

# Plot parsed PMBW data
plt.figure(figsize=(12, 6))

# Plot lines and fills in reverse order of thread count
for i, nthreads in reversed(list(enumerate(thread_counts))):
    group = df[df['nthreads'] == nthreads]
    color = colors[i]
    plt.plot(
        group['log2_areasize'],
        group['bandwidth'],
        label=f'p={nthreads}',
        marker='x',
        markersize=4,
        linewidth=0.8,
        linestyle=':',
        color=color
    )
    plt.fill_between(
        group['log2_areasize'],
        group['bandwidth'],
        color=color,
        alpha=0.2
    )

plt.title("PMBW Parsed Data")
plt.xlabel("Array Size log2 [B]")
plt.ylabel("Bandwidth [GiB/s]")
plt.legend(title="Threads", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='both', linestyle=':', linewidth=0.5)
plt.xlim(left=9, right=30)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
plt.tight_layout()
plt.show()
