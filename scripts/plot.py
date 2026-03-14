import matplotlib.pyplot as plt
import numpy as np

# Classification report data
classes = [0, 1, 2, 3, 4]
precision = [0.65, 0.33, 0.36, 0.00, 0.50]
recall = [0.61, 0.20, 0.57, 0.00, 0.29]
f1_score = [0.63, 0.25, 0.44, 0.00, 0.36]
support = [18, 5, 7, 0, 7]

accuracy = 0.49
macro_avg = [0.37, 0.33, 0.34]  # precision, recall, f1
weighted_avg = [0.52, 0.49, 0.49]  # precision, recall, f1

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Classification Report Statistics', fontsize=16)

# Bar plot for precision, recall, f1-score per class
x = np.arange(len(classes))
width = 0.25

axes[0, 0].bar(x - width, precision, width, label='Precision', color='skyblue')
axes[0, 0].bar(x, recall, width, label='Recall', color='lightgreen')
axes[0, 0].bar(x + width, f1_score, width, label='F1-Score', color='salmon')
axes[0, 0].set_title('Metrics per Class')
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(classes)
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1)

# Bar plot for support
axes[0, 1].bar(classes, support, color='orange')
axes[0, 1].set_title('Support per Class')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Support Count')
axes[0, 1].set_xticks(classes)

# Overall metrics comparison
metrics = ['Precision', 'Recall', 'F1-Score']
macro_vals = macro_avg
weighted_vals = weighted_avg

x_metrics = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x_metrics - width/2, macro_vals, width, label='Macro Avg', color='purple')
axes[1, 0].bar(x_metrics + width/2, weighted_vals, width, label='Weighted Avg', color='teal')
axes[1, 0].set_title('Overall Averages')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_xticks(x_metrics)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

# Accuracy display
axes[1, 1].text(0.5, 0.5, f'Accuracy: {accuracy:.2f}', fontsize=20, ha='center', va='center')
axes[1, 1].set_title('Overall Accuracy')
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
