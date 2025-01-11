import os
import json
import sys

def parse_results(results_dir):
    results = {}

    # Parse likwid results
    for filename in os.listdir(results_dir):
        if 'likwid-bench-' in filename and not filename.endswith('.log.pbmw'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                for line in f:
                    if 'MByte/s:' in line:
                        # Extract the specific likwid test name
                        test_name = filename.split('likwid-bench-')[1].replace('.log', '').replace('_avx512', '').replace('_mem', '').replace(results_dir + '/', '')
                        
                        # Extract MB/s value and convert to GiB/s
                        value_mb = float(line.split(':')[1].strip())
                        value_gib = value_mb / 1024
                        results[f'likwid_{test_name}'] = value_gib

    # Parse sysbench results
    sysbench_file = os.path.join(results_dir, 'sysbench-avg.log')
    if os.path.exists(sysbench_file):
        with open(sysbench_file, 'r') as f:
            for line in f:
                if 'Average throughput' in line:
                    # Extract the value in MiB/sec and convert to GiB/sec
                    value_str = line.split(':')[1].strip().split()[0]
                    value_mib = float(value_str)
                    value_gib = value_mib / 1024
                    results['sysbench_memory_read_gib'] = value_gib

    # Parse llama.cpp results
    llama_file = os.path.join(results_dir, 'llama-bench.json')
    if os.path.exists(llama_file):
        with open(llama_file, 'r') as f:
            llama_data = json.load(f)
            for entry in llama_data:
                model_name = entry['model_filename']
                tokens_per_second = entry['avg_ts']
                model_size_bytes = entry['model_size']
                model_size_gb = model_size_bytes / 1024 / 1024 / 1024
                mbw = tokens_per_second * model_size_gb
                results[f'llama_{model_name.replace(".gguf", "").replace("/", "_")}'] = {
                    'tokens_per_second': tokens_per_second,
                    'model_size_gb': model_size_gb,
                    'mbw': mbw
                }

    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_results.py <results_directory>")
        sys.exit(1)
    results_dir = sys.argv[1]
    parsed_results = parse_results(results_dir)
    print(f"Results for {results_dir}:")
    print(json.dumps(parsed_results, indent=4))

if __name__ == "__main__":
    main()
