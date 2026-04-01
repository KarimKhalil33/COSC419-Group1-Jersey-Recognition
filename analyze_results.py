import json
import collections
import matplotlib.pyplot as plt
import numpy as np

with open('test_smart_pred_final.txt', 'r') as f:
    predictions = json.load(f)

with open('data/SoccerNet/jersey-2023/test/test/test_gt.json', 'r') as f:
    ground_truth = json.load(f)

# Overall accuracy
correct = 0
total = 0
for tracklet_id, gt_number in ground_truth.items():
    pred_number = predictions.get(str(tracklet_id), -1)
    if pred_number == gt_number:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"Overall accuracy: {correct}/{total} = {accuracy:.1f}%")

# Illegible stats
num_illegible_pred = sum(1 for v in predictions.values() if v == -1)
num_illegible_gt = sum(1 for v in ground_truth.values() if v == -1)
print(f"\nTracklets predicted illegible (-1): {num_illegible_pred}")
print(f"Tracklets actually illegible (-1) in GT: {num_illegible_gt}")

# Per-number accuracy: which jersey numbers are hardest to predict?
number_correct = collections.defaultdict(int)
number_total = collections.defaultdict(int)

for tracklet_id, gt_number in ground_truth.items():
    pred_number = predictions.get(str(tracklet_id), -1)
    number_total[gt_number] += 1
    if pred_number == gt_number:
        number_correct[gt_number] += 1

print("\nPer-number accuracy (numbers with 5+ tracklets):")
results_by_number = []
for number, total_count in number_total.items():
    if total_count >= 5:
        acc = number_correct[number] / total_count * 100
        results_by_number.append((number, number_correct[number], total_count, acc))

results_by_number.sort(key=lambda x: x[3])  # sort by accuracy ascending
for number, correct_count, total_count, acc in results_by_number:
    print(f"  #{number:>2}: {correct_count}/{total_count} = {acc:.0f}%")

# Most common mistakes: what does the model predict instead of the correct number?
print("\nMost common mistakes (excluding illegible):")
mistakes = collections.Counter()
for tracklet_id, gt_number in ground_truth.items():
    pred_number = predictions.get(str(tracklet_id), -1)
    if pred_number != gt_number and gt_number != -1 and pred_number != -1:
        mistakes[(gt_number, pred_number)] += 1

for (gt, pred), count in mistakes.most_common(10):
    print(f"  GT={gt} predicted as {pred}: {count} times")

# --- Plot 1: Accuracy by jersey number ---
plot_numbers = [x[0] for x in results_by_number]
plot_accs = [x[3] for x in results_by_number]
avg_acc = sum(plot_accs) / len(plot_accs)
colors = ['green' if a >= avg_acc else 'red' for a in plot_accs]

plt.figure(figsize=(14, 5))
plt.bar(range(len(plot_numbers)), plot_accs, color=colors)
plt.axhline(avg_acc, color='black', linestyle='--', linewidth=1.2, label=f'Average ({avg_acc:.0f}%)')
plt.xticks(range(len(plot_numbers)), [f'#{n}' for n in plot_numbers], rotation=90)
plt.ylabel('Accuracy (%)')
plt.title('Per-Number Accuracy (green = above average, red = below average)')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_by_number.png', dpi=150)
print("\nPlot saved to accuracy_by_number.png")

# --- Plot 2: Predicted vs actual distribution ---
gt_counter = collections.Counter(v for v in ground_truth.values() if v != -1)
pred_counter = collections.Counter(v for v in predictions.values() if v != -1)
all_numbers = sorted(set(gt_counter.keys()) | set(pred_counter.keys()))

gt_counts = [gt_counter.get(n, 0) for n in all_numbers]
pred_counts = [pred_counter.get(n, 0) for n in all_numbers]
x = np.arange(len(all_numbers))
width = 0.4

plt.figure(figsize=(16, 5))
plt.bar(x - width/2, gt_counts, width, label='Ground truth', color='steelblue', alpha=0.8)
plt.bar(x + width/2, pred_counts, width, label='Predictions', color='orange', alpha=0.8)
plt.xticks(x, [f'#{n}' for n in all_numbers], rotation=90)
plt.ylabel('Number of tracklets')
plt.title('Predicted vs Actual Jersey Number Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual_distribution.png', dpi=150)
print("Plot saved to predicted_vs_actual_distribution.png")

# --- Plot 3: Prediction outcome breakdown (pie chart) ---
# For every tracklet there are exactly 4 possible outcomes:
correct_prediction = 0   # GT has a number, model got it right
wrong_number = 0         # GT has a number, model predicted the wrong number
wrongly_illegible = 0    # GT has a number, model said -1 (gave up unnecessarily)
wrongly_legible = 0      # GT=-1 (not visible), model predicted a number anyway
correctly_illegible = 0  # GT=-1, model also said -1 (correct abstention)

for tracklet_id, gt_number in ground_truth.items():
    pred_number = predictions.get(str(tracklet_id), -1)
    if gt_number == -1 and pred_number == -1:
        correctly_illegible += 1
    elif gt_number == -1 and pred_number != -1:
        wrongly_legible += 1
    elif gt_number != -1 and pred_number == gt_number:
        correct_prediction += 1
    elif gt_number != -1 and pred_number == -1:
        wrongly_illegible += 1
    else:
        wrong_number += 1

total_correct = correct_prediction + correctly_illegible
labels = [
    f'Correct ({total_correct})',
    f'Wrong number ({wrong_number})',
    f'Wrongly illegible ({wrongly_illegible})',
    f'Wrongly legible ({wrongly_legible})',
]
sizes = [total_correct, wrong_number, wrongly_illegible, wrongly_legible]
colors_pie = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=140)
plt.title('Prediction Outcome Breakdown\n(all tracklets in test set)')
plt.tight_layout()
plt.savefig('prediction_breakdown.png', dpi=150)
print("Plot saved to prediction_breakdown.png")

print(f"\nOutcome summary:")
print(f"  Correct prediction:    {correct_prediction}")
print(f"  Correctly illegible:   {correctly_illegible}")
print(f"  Wrong number:          {wrong_number}")
print(f"  Wrongly illegible:     {wrongly_illegible}  <- model gave up when it shouldn't have")
print(f"  Wrongly legible:       {wrongly_legible}  <- model guessed when it shouldn't have")
