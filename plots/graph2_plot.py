import matplotlib.pyplot as plt
import os

# Define cache sizes in ascending order
cache_sizes = [32, 64, 128, 256]

# Matrix Multiplication Miss Rates (%)
matrix_miss_rates = {
    'LRU': [4.58386, 3.25387, 2.87057, 2.46772],
    'LFU': [59.2873, 67.9839, 71.5978, 65.7903],
    'MRU': [34.6939, 47.3784, 45.4415, 42.4233],
    'Mockingjay': [8.90466, 7.18327, 4.59351, 2.17513],
}

# Insertion Sort Miss Rates (%)
insertion_miss_rates = {
    'LRU': [0.0258398, 0.0125641, 0.00609757, 0.00305027],
    'LFU': [3.33333, 3.22896, 3.12805, 3.12653],
    'MRU': [3.33333, 3.22896, 3.12805, 3.12653],
    'Mockingjay': [3.33333, 3.22896, 3.12805, 3.12653],
}

# DFT Miss Rates (%)
dft_miss_rates = {
    'LRU': [12.9408, 12.7201, 12.6099, 12.5061],
    'LFU': [42.9292, 42.3862, 42.3296, 41.2473],
    'MRU': [4.0856, 2.43665, 1.21951, 0.414837],
    'Mockingjay': [2.63862, 1.56022, 0.879573, 0.479464],
}

# FFT Miss Rates (%)
fft_miss_rates = {
    'LRU': [30.8424, 24.384, 18.9857, 15.8051],
    'LFU': [60.1223, 52.8395, 38.9076, 34.9426],
    'MRU': [77.1399, 77.1935, 77.2966, 75.5096],
    'Mockingjay': [60.462, 56.9712, 53.1789, 58.6548],
}

# Trace Miss Rates (gcc.txt) (%)
trace_miss_rates = {
    'LRU': [9.471, 5.411, 2.798, 1.98],
    'LFU': [31.821, 22.124, 13.1, 6.818],
    'MRU': [51.718, 45.35, 37.003, 21.661],
    'Mockingjay': [30.847, 20.933, 12.721, 4.787],
}

# Define a directory to save plots
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# Line styles for better differentiation
line_styles = {
    'LRU': 'o-',
    'LFU': 's--',
    'MRU': '^-',
    'Mockingjay': 'd-.',
}

# Function to plot and save miss rates for a given algorithm
def plot_and_save(cache_sizes, miss_rates, algorithm_name):
    plt.figure(figsize=(10, 6))
    for policy, rates in miss_rates.items():
        plt.plot(cache_sizes, rates, line_styles.get(policy, 'o-'), label=policy, markersize=8, linewidth=2)
    
    plt.title(f'{algorithm_name} Miss Rates vs. Cache Size', fontsize=16)
    plt.xlabel('Cache Size (Number of Cache Lines)', fontsize=14)
    plt.ylabel('Miss Rate (%)', fontsize=14)
    plt.xticks(cache_sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Cache Replacement Policy', fontsize=12)
    
    # Generate a filename by replacing spaces and converting to lowercase
    safe_algorithm_name = algorithm_name.replace(' ', '_').lower()
    filename = os.path.join(plot_dir, f"{safe_algorithm_name}_miss_rates.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    # Close the plot to free memory
    plt.close()

# Plotting for each algorithm
plot_and_save(cache_sizes, matrix_miss_rates, 'Matrix Multiplication')
plot_and_save(cache_sizes, insertion_miss_rates, 'Insertion Sort')
plot_and_save(cache_sizes, dft_miss_rates, 'DFT')
plot_and_save(cache_sizes, fft_miss_rates, 'FFT')
plot_and_save(cache_sizes, trace_miss_rates, 'Trace File (gcc.txt)')
