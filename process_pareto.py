import csv
import sys
import argparse
import statistics
from collections import defaultdict

def calc_stats(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mn = min(values)
    mx = max(values)
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mn, avg, mx, std

def main():
    parser = argparse.ArgumentParser(description="Generate pareto report from CSV")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()

    params = [
        "dtype", "dst_dtype", "hidden_size", "num_tokens", "topk", 
        "world_size", "rank", "num_shared_experts", "dp_size", "sp_size", 
        "num_experts", "ep_size"
    ]

    data_by_family_op = defaultdict(list)

    try:
        with open(args.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fam = row.get("report_family", "Unknown")
                op = row.get("op_name", "Unknown")
                
                try:
                    lat = float(row["rel_latency"])
                    bw = float(row["rel_mem_bw"])
                except (ValueError, KeyError):
                    continue
                
                data_by_family_op[(fam, op)].append((row, lat, bw))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    for (fam, op), records in data_by_family_op.items():
        all_lats = [r[1] for r in records]
        all_bws = [r[2] for r in records]
        
        g_l_min, g_l_avg, g_l_max, g_l_std = calc_stats(all_lats)
        g_b_min, g_b_avg, g_b_max, g_b_std = calc_stats(all_bws)
        
        print("\n" + "=" * 140)
        print(f"REPORT FAMILY: {fam} | OP NAME: {op}")
        print("=" * 140)
        print(f"{'PARAMETER':<20} {'VALUE':<15} {'REL LATENCY [Min / Avg / Max / Std]':<45} | {'REL BW [Min / Avg / Max / Std]'}")
        print("-" * 140)
        print(f"{'Global Average':<20} {'-':<15} {g_l_min:<8.2f}/ {g_l_avg:<8.2f}/ {g_l_max:<8.2f}/ {g_l_std:<8.2f}     | {g_b_min:<8.2f}/ {g_b_avg:<8.2f}/ {g_b_max:<8.2f}/ {g_b_std:<8.2f}")
        print("-" * 140)
        
        for param in params:
            grouped = defaultdict(list)
            for row, lat, bw in records:
                val = row.get(param, "N/A")
                grouped[val].append((lat, bw))
                
            if not grouped or all(k == "N/A" for k in grouped.keys()):
                continue
                
            print(f"{param.upper().replace('_', ' ')}")
            
            # Sort by float if possible, else string
            sorted_vals = sorted(grouped.keys(), key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
            
            for val in sorted_vals:
                group_lats = [item[0] for item in grouped[val]]
                group_bws = [item[1] for item in grouped[val]]
                
                l_min, l_avg, l_max, l_std = calc_stats(group_lats)
                b_min, b_avg, b_max, b_std = calc_stats(group_bws)
                
                print(f"{'':<20} {val:<15} {l_min:<8.2f}/ {l_avg:<8.2f}/ {l_max:<8.2f}/ {l_std:<8.2f}     | {b_min:<8.2f}/ {b_avg:<8.2f}/ {b_max:<8.2f}/ {b_std:<8.2f}")
                
        print("=" * 140 + "\n")

if __name__ == "__main__":
    main()