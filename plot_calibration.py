import numpy as np
import matplotlib.pyplot as plt

# These are the empirical calibration points hardcoded in helpers.py
# pred_conf: what the model outputs as its confidence
# true_conf: what the true probability actually is at that confidence level
pred_conf = [0.27909853, 0.36124473, 0.45841138, 0.54969843, 0.65061644, 0.74988696, 0.85455219, 0.99713011]
true_conf = [0.12, 0.13609467, 0.25098814, 0.38429752, 0.43631613, 0.48029819, 0.56463068, 0.94912186]

# A perfectly calibrated model would lie on the diagonal (y=x)
# i.e. if the model says 80% confident, it should be right 80% of the time
perfect = [0, 1]

plt.figure(figsize=(7, 7))

# Plot perfect calibration diagonal
plt.plot(perfect, perfect, 'k--', label='Perfect calibration (y=x)', linewidth=1.5)

# Plot the actual calibration curve
plt.plot(pred_conf, true_conf, 'o-', color='steelblue', linewidth=2, markersize=8, label='Model calibration')

# Shade the gap between model and perfect to make overconfidence visible
plt.fill_between(pred_conf, pred_conf, true_conf, alpha=0.15, color='red', label='Overconfidence gap')

plt.xlabel('Model confidence (what the model thinks)', fontsize=12)
plt.ylabel('True probability (how often it is actually correct)', fontsize=12)
plt.title('PARSeq Calibration Curve\nHow overconfident is the model?', fontsize=13)
plt.legend(fontsize=11)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_curve.png', dpi=150)
print("Plot saved to calibration_curve.png")

# Print a summary of the overconfidence at each point
print("\nCalibration summary:")
print(f"{'Model confidence':>20} {'True probability':>18} {'Gap (overconfidence)':>22}")
print("-" * 62)
for pred, true in zip(pred_conf, true_conf):
    gap = pred - true
    print(f"{pred:>20.3f} {true:>18.3f} {gap:>+22.3f}")

print("\nPositive gap = model is overconfident (claims more certainty than warranted)")
print("Negative gap = model is underconfident")
avg_gap = np.mean([p - t for p, t in zip(pred_conf, true_conf)])
print(f"Average gap: {avg_gap:+.3f}")
