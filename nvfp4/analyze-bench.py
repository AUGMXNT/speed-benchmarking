#!/usr/bin/env python3

import json
import os
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def load_benchmark_data():
    """Load all benchmark JSON files and organize by concurrency level."""
    data_by_concurrency = defaultdict(list)

    # Scan all directories for JSON files
    for folder in Path('.').iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            config_name = folder.name
            for json_file in folder.glob('*concurrency*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        # Extract concurrency from filename
                        concurrency = data.get('max_concurrency')
                        if concurrency:
                            data['config'] = config_name
                            data_by_concurrency[concurrency].append(data)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not parse {json_file}: {e}[/yellow]")

    return data_by_concurrency

def format_value(value, is_best, lower_is_better=False):
    """Format a value with color highlighting if it's the best."""
    if value is None:
        return "[dim]N/A[/dim]"

    formatted = f"{value:.2f}"
    if is_best:
        return f"[bold green]{formatted}[/bold green]"
    return formatted

def find_best_values(configs, lower_is_better_metrics):
    """Find the best value for each metric across all configs."""
    best = {}
    metrics = [
        'request_throughput', 'prefill_throughput', 'output_throughput',
        'total_token_throughput', 'max_output_tokens_per_s',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
        'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms'
    ]

    for metric in metrics:
        values = [c.get(metric) for c in configs if c.get(metric) is not None]
        if values:
            if metric in lower_is_better_metrics:
                best[metric] = min(values)
            else:
                best[metric] = max(values)

    return best

def create_comparison_table(concurrency, configs):
    """Create a comparison table for a specific concurrency level."""
    lower_is_better = {'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
                       'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms'}

    # Sort configs by name for consistent ordering
    configs = sorted(configs, key=lambda x: x['config'])

    # Calculate prefill throughput for each config
    for config in configs:
        if config.get('total_token_throughput') and config.get('output_throughput'):
            config['prefill_throughput'] = config['total_token_throughput'] - config['output_throughput']

    # Find best values for highlighting
    best = find_best_values(configs, lower_is_better)

    # Create table
    table = Table(
        title=f"[bold cyan]Concurrency: {concurrency}[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1)
    )

    # Add columns
    table.add_column("Config", style="cyan", no_wrap=True)
    table.add_column("Req/s", justify="right")
    table.add_column("Prefill\nTok/s", justify="right")
    table.add_column("Decode\nTok/s", justify="right")
    table.add_column("Total\nTok/s", justify="right")
    table.add_column("Max Out\nTok/s", justify="right")
    table.add_column("TTFT\nmean", justify="right")
    table.add_column("TTFT\nmed", justify="right")
    table.add_column("TTFT\np99", justify="right")
    table.add_column("TPOT\nmean", justify="right")
    table.add_column("TPOT\nmed", justify="right")
    table.add_column("TPOT\np99", justify="right")

    # Add rows
    for config in configs:
        table.add_row(
            config['config'],
            format_value(
                config.get('request_throughput'),
                config.get('request_throughput') == best.get('request_throughput')
            ),
            format_value(
                config.get('prefill_throughput'),
                config.get('prefill_throughput') == best.get('prefill_throughput')
            ),
            format_value(
                config.get('output_throughput'),
                config.get('output_throughput') == best.get('output_throughput')
            ),
            format_value(
                config.get('total_token_throughput'),
                config.get('total_token_throughput') == best.get('total_token_throughput')
            ),
            format_value(
                config.get('max_output_tokens_per_s'),
                config.get('max_output_tokens_per_s') == best.get('max_output_tokens_per_s')
            ),
            format_value(
                config.get('mean_ttft_ms'),
                config.get('mean_ttft_ms') == best.get('mean_ttft_ms'),
                lower_is_better=True
            ),
            format_value(
                config.get('median_ttft_ms'),
                config.get('median_ttft_ms') == best.get('median_ttft_ms'),
                lower_is_better=True
            ),
            format_value(
                config.get('p99_ttft_ms'),
                config.get('p99_ttft_ms') == best.get('p99_ttft_ms'),
                lower_is_better=True
            ),
            format_value(
                config.get('mean_tpot_ms'),
                config.get('mean_tpot_ms') == best.get('mean_tpot_ms'),
                lower_is_better=True
            ),
            format_value(
                config.get('median_tpot_ms'),
                config.get('median_tpot_ms') == best.get('median_tpot_ms'),
                lower_is_better=True
            ),
            format_value(
                config.get('p99_tpot_ms'),
                config.get('p99_tpot_ms') == best.get('p99_tpot_ms'),
                lower_is_better=True
            ),
        )

    return table

def format_value_markdown(value, is_best, lower_is_better=False):
    """Format a value for markdown with bold highlighting if it's the best."""
    if value is None:
        return "N/A"

    formatted = f"{value:.2f}"
    if is_best:
        return f"**{formatted}**"
    return formatted

def create_markdown_table(concurrency, configs):
    """Create a markdown table for a specific concurrency level."""
    lower_is_better = {'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
                       'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms'}

    # Sort configs by name for consistent ordering
    configs = sorted(configs, key=lambda x: x['config'])

    # Calculate prefill throughput for each config
    for config in configs:
        if config.get('total_token_throughput') and config.get('output_throughput'):
            config['prefill_throughput'] = config['total_token_throughput'] - config['output_throughput']

    # Find best values for highlighting
    best = find_best_values(configs, lower_is_better)

    # Create markdown table
    lines = []
    lines.append(f"### Concurrency: {concurrency}")
    lines.append("")

    # Header
    header = "| Config | Req/s | Prefill Tok/s | Decode Tok/s | Total Tok/s | Max Out Tok/s | TTFT mean | TTFT med | TTFT p99 | TPOT mean | TPOT med | TPOT p99 |"
    separator = "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for config in configs:
        row = [
            config['config'],
            format_value_markdown(
                config.get('request_throughput'),
                config.get('request_throughput') == best.get('request_throughput')
            ),
            format_value_markdown(
                config.get('prefill_throughput'),
                config.get('prefill_throughput') == best.get('prefill_throughput')
            ),
            format_value_markdown(
                config.get('output_throughput'),
                config.get('output_throughput') == best.get('output_throughput')
            ),
            format_value_markdown(
                config.get('total_token_throughput'),
                config.get('total_token_throughput') == best.get('total_token_throughput')
            ),
            format_value_markdown(
                config.get('max_output_tokens_per_s'),
                config.get('max_output_tokens_per_s') == best.get('max_output_tokens_per_s')
            ),
            format_value_markdown(
                config.get('mean_ttft_ms'),
                config.get('mean_ttft_ms') == best.get('mean_ttft_ms'),
                lower_is_better=True
            ),
            format_value_markdown(
                config.get('median_ttft_ms'),
                config.get('median_ttft_ms') == best.get('median_ttft_ms'),
                lower_is_better=True
            ),
            format_value_markdown(
                config.get('p99_ttft_ms'),
                config.get('p99_ttft_ms') == best.get('p99_ttft_ms'),
                lower_is_better=True
            ),
            format_value_markdown(
                config.get('mean_tpot_ms'),
                config.get('mean_tpot_ms') == best.get('mean_tpot_ms'),
                lower_is_better=True
            ),
            format_value_markdown(
                config.get('median_tpot_ms'),
                config.get('median_tpot_ms') == best.get('median_tpot_ms'),
                lower_is_better=True
            ),
            format_value_markdown(
                config.get('p99_tpot_ms'),
                config.get('p99_tpot_ms') == best.get('p99_tpot_ms'),
                lower_is_better=True
            ),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def update_readme_with_results(markdown_tables):
    """Update README.md with the benchmark results in a Results section."""
    readme_path = Path('README.md')

    # Read current README
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
    else:
        console.print("[red]README.md not found![/red]")
        return

    # Create the results section
    results_section = "## Results\n\n"
    results_section += "*Generated by analyze-bench.py - Best values shown in **bold***\n\n"
    results_section += markdown_tables

    # Markers for idempotent updates
    start_marker = "<!-- RESULTS_START -->"
    end_marker = "<!-- RESULTS_END -->"

    # Check if markers exist
    if start_marker in content and end_marker in content:
        # Replace existing results section
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker) + len(end_marker)
        new_content = (
            content[:start_idx] +
            start_marker + "\n" +
            results_section + "\n" +
            end_marker +
            content[end_idx:]
        )
    else:
        # Append new results section at the end
        new_content = content.rstrip() + "\n\n" + start_marker + "\n" + results_section + "\n" + end_marker + "\n"

    # Write back to README
    with open(readme_path, 'w') as f:
        f.write(new_content)

    console.print("[green]✓ README.md updated with benchmark results[/green]")

def main():
    """Main function to analyze and display benchmark comparisons."""
    data_by_concurrency = load_benchmark_data()

    if not data_by_concurrency:
        console.print("[red]No benchmark data found![/red]")
        return

    # Sort by concurrency level
    sorted_concurrencies = sorted(data_by_concurrency.keys())

    console.print("\n[bold]Benchmark Analysis - Best values highlighted in [green]green[/green][/bold]\n")
    console.print("[dim]Throughput metrics: higher is better | Latency metrics (TTFT/TPOT): lower is better[/dim]\n")

    # Generate both console tables and markdown tables
    markdown_tables = []

    for concurrency in sorted_concurrencies:
        configs = data_by_concurrency[concurrency]
        # Skip concurrency levels with only 1 config (nothing to compare)
        if len(configs) <= 1:
            continue

        # Display console table
        table = create_comparison_table(concurrency, configs)
        console.print(table)
        console.print()

        # Generate markdown table
        md_table = create_markdown_table(concurrency, configs)
        markdown_tables.append(md_table)

    # Update README.md with all markdown tables
    if markdown_tables:
        all_markdown = "\n\n".join(markdown_tables)
        update_readme_with_results(all_markdown)

if __name__ == "__main__":
    main()
