#!/usr/bin/env python3
import os
import re

input_path = "/Users/marcher/Desktop/Marcher_code/AI量化交易高效构建交易策略的新路径/full.md"
output_dir = "/Users/marcher/Desktop/Marcher_code/AI量化交易高效构建交易策略的新路径/full_md_parts"

os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_file = None
buffer = []
for line in lines:
    if line.startswith("#"):
        # Write previous buffer if any
        if current_file is not None and buffer:
            # Clean filename
            safe_name = re.sub(r"[\\/*?:\"<>|]", "", current_file).strip().replace(" ", "_") + ".md"
            out_path = os.path.join(output_dir, safe_name)
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("".join(buffer).strip() + "\n")
        # Start new file
        current_file = line.lstrip("#").strip()
        buffer = [line]
    else:
        buffer.append(line)

# Write the last section
if current_file is not None and buffer:
    safe_name = re.sub(r"[\\/*?:\"<>|]", "", current_file).strip().replace(" ", "_") + ".md"
    out_path = os.path.join(output_dir, safe_name)
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("".join(buffer).strip() + "\n")

print(f"Split completed. Files are in {output_dir}")
