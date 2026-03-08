import json
from collections import Counter

pred = json.load(open("/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/proposed_pipeline/runs_final_pipeline/test_new_pred_final.json"))
gt   = json.load(open("/scratch/st-li1210-1/yijun/519Project/jersey-number-pipeline/data/SoccerNet/jersey-2023/test/test_gt.json"))

keys = sorted(set(pred.keys()) & set(gt.keys()), key=lambda x:int(x))
n = len(keys)

acc = sum(1 for k in keys if pred[k] == gt[k]) / n

pred_vals = [pred[k] for k in keys]
gt_vals   = [gt[k] for k in keys]

pred_m1 = sum(1 for v in pred_vals if v == -1) / n
gt_m1   = sum(1 for v in gt_vals if v == -1) / n

cp = Counter(pred_vals)
cg = Counter(gt_vals)

print("N:", n)
print("Acc:", acc)
print("Pred -1 rate:", pred_m1, "GT -1 rate:", gt_m1)

print("\nTop-10 predicted:", cp.most_common(10))
print("Top-10 GT:", cg.most_common(10))

pred_set = set(pred_vals)
missing_gt_labels = sorted([v for v in set(gt_vals) if v not in pred_set and v != -1])
print("\n#GT labels never predicted (excluding -1):", len(missing_gt_labels))
print("Example missing labels:", missing_gt_labels[:30])