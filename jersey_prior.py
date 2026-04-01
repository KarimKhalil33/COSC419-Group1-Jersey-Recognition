import json
import collections
import matplotlib.pyplot as plt
import numpy as np

# Load training ground truth
with open('data/SoccerNet/jersey-2023/train/train/train_gt.json', 'r') as f:
    train_gt = json.load(f)

# Count frequency of each jersey number (exclude illegible -1)
counter = collections.Counter()
for tracklet_id, number in train_gt.items():
    if number != -1:
        counter[number] += 1

total = sum(counter.values())
print(f"Total legible tracklets in training set: {total}")
print(f"Total tracklets (including illegible): {len(train_gt)}")
print(f"Illegible tracklets: {len(train_gt) - total}")

# Single digit vs double digit split
single_digit = sum(counter[n] for n in counter if 1 <= n <= 9)
double_digit = sum(counter[n] for n in counter if 10 <= n <= 99)
print(f"\nSingle digit (1-9): {single_digit} tracklets = {single_digit/total*100:.1f}%")
print(f"Double digit (10-99): {double_digit} tracklets = {double_digit/total*100:.1f}%")

# The hardcoded bias in helpers.py
# bias_for_digits = [0.06, 0.094, 0.094, ..., 0.094]
# Index 0 = end token (not a number), indices 1-10 = digits 0-9
# Single digit prior = 0.06, each double digit token = 0.094
# This implies single digits are LESS likely than double digits
hardcoded_single = 0.06
hardcoded_double = 0.094
print(f"\nHardcoded bias in helpers.py:")
print(f"  Single digit weight: {hardcoded_single}")
print(f"  Double digit token weight: {hardcoded_double}")
print(f"  Implies single digits are {hardcoded_single/hardcoded_double:.2f}x as likely as double digit tokens")

# Most common jersey numbers
print("\nTop 15 most common jersey numbers in training data:")
for number, count in counter.most_common(15):
    bar = '#' * int(count / total * 500)
    print(f"  #{number:>2}: {count:>4} ({count/total*100:.1f}%) {bar}")

# Data-derived prior: what should the bias actually be?
print("\nData-derived frequency per number (top 20):")
sorted_numbers = sorted(counter.keys())
for number in sorted_numbers[:20]:
    freq = counter[number] / total
    print(f"  #{number:>2}: {freq:.4f}")

# Compare: is single digit actually less common than the bias assumes?
print("\n--- Summary ---")
print(f"In the actual training data, single digit numbers make up {single_digit/total*100:.1f}% of tracklets.")
print(f"The hardcoded bias weights single digits LOWER than double digit tokens ({hardcoded_single} vs {hardcoded_double}).")
if single_digit / total > 0.5:
    print("The data suggests single digit numbers are actually MORE common — the bias may be correct in direction.")
else:
    print("The data confirms single digit numbers are less common than double digit ones.")

# Plot distribution
numbers = sorted(counter.keys())
counts = [counter[n] for n in numbers]
freqs = [c / total for c in counts]

plt.figure(figsize=(14, 5))
plt.bar(numbers, freqs, color='steelblue')
plt.xlabel('Jersey Number')
plt.ylabel('Frequency in Training Set')
plt.title('Jersey Number Distribution in SoccerNet Training Data')
plt.xticks(numbers, rotation=90)
plt.tight_layout()
plt.savefig('jersey_number_distribution.png', dpi=150)
print("\nPlot saved to jersey_number_distribution.png")
