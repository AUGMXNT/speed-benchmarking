import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from scipy.interpolate import make_interp_spline

def load_and_parse_pmbw_stats(file_path):
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
    return pd.DataFrame(data)

# Load data from both files
file_path_nps1 = 'results-nps1/pmbw-stats.txt'
df_nps1 = load_and_parse_pmbw_stats(file_path_nps1)
df_nps1['log2_areasize'] = np.log2(df_nps1['areasize'])

file_path_nps4 = 'results-nps4-srat-l3/pmbw-stats.txt'
df_nps4 = load_and_parse_pmbw_stats(file_path_nps4)
df_nps4['log2_areasize'] = np.log2(df_nps4['areasize'])

# Filter for p=32
df_nps1_p32 = df_nps1[df_nps1['nthreads'] == 32]
df_nps4_p32 = df_nps4[df_nps4['nthreads'] == 32]

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

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(
    df_nps1_p32['log2_areasize'],
    df_nps1_p32['bandwidth'],
    label='results-nps1 p=32',
    marker='x',
    markersize=4,
    linewidth=1,
    linestyle='-',
    color='red'
)

plt.plot(
    df_nps4_p32['log2_areasize'],
    df_nps4_p32['bandwidth'],
    label='results-nps4 p=32',
    marker='o',
    markersize=4,
    linewidth=1,
    linestyle=':',
    color='blue'
)

plt.title("PMBW Comparison (p=32)")
plt.xlabel("Array Size log2 [B]")
plt.ylabel("Bandwidth [GiB/s]")
plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='both', linestyle=':', linewidth=0.5)
plt.xlim(left=10, right=30)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
plt.tight_layout()
plt.show()