import os
import json

with open('data.json', 'r') as f:
        data = json.load(f)

# Initialize counters
total_objects = len(data)
non_interruption = 0
interruption = 0

for obj in data:
    if obj['classification'] == 'non-interruption':
        non_interruption += 1
    elif obj['classification'] == 'interruption':
        interruption += 1

# Calculate percentages
non_interruption_pct = (non_interruption / total_objects) * 100
interruption_pct = (interruption / total_objects) * 100

# Print results
print(f'Count: {total_objects}')
print(f'True Interruption: {interruption} ({interruption_pct:.1f}%)')
print(f'False Interruption: {non_interruption} ({non_interruption_pct:.1f}%)')