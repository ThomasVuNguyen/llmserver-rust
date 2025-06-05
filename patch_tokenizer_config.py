#!/usr/bin/env python3
import json
import os
import sys
import shutil

# Path to the model's tokenizer_config.json
model_path = "/home/thomas/.cache/huggingface/hub/models--ThomasTheMaker--Qwen3-1.7B-RKLLM-v1.2.0/snapshots/47f5a43b9bca8e169288ed19ac13e44438994414"
tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
blob_path = os.path.realpath(tokenizer_config_path)

print(f"Tokenizer config symlinks to: {blob_path}")

# Create backup of original file
backup_path = blob_path + ".backup"
if not os.path.exists(backup_path):
    print(f"Creating backup at: {backup_path}")
    shutil.copy2(blob_path, backup_path)
else:
    print(f"Backup already exists at: {backup_path}")

# Read the current config
with open(blob_path, 'r') as f:
    config = json.load(f)

# Add the missing legacy field
if 'legacy' not in config:
    config['legacy'] = True
    print("Added 'legacy': true to config")
else:
    print("Config already has 'legacy' field with value:", config['legacy'])

# Print all keys and line count for debugging
print(f"Config keys: {list(config.keys())}")
print(f"Line count (approx): {len(json.dumps(config, indent=2).splitlines())}")

# Write the updated config back
with open(blob_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated tokenizer config saved to: {blob_path}")
print("Done! Try running your LLM server now.")
