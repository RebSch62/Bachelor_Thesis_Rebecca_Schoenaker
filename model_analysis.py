import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

preds  = np.load("best_preds.npy")
labels = np.load("best_labels.npy")

tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

print(f"True Positives  (tremor correctly detected):     {tp}")
print(f"True Negatives  (non-tremor correctly rejected): {tn}")
print(f"False Positives (non-tremor called tremor):      {fp}")
print(f"False Negatives (tremor missed):                 {fn}")
print(f"\nSensitivity: {tp/(tp+fn):.3f}")  # are we catching tremor?
print(f"Specificity: {tn/(tn+fp):.3f}")   # are we avoiding false alarms?
print(f"\n{classification_report(labels, preds, target_names=['non-tremor', 'tremor'])}")