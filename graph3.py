import matplotlib.pyplot as plt
import numpy as np

datasets = ['Math', 'Commonsense', 'Multi-hop']
standard_f1 = [0.158, 0.697, 0.072]
zero_shot_f1 = [0.167, 0.359, 0.130]
few_shot_f1 = [0.913, 0.846, 0.356]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, standard_f1, width, label='Standard F1', color='#4472C4')
bars2 = ax.bar(x, zero_shot_f1, width, label='Zero-shot CoT F1', color='#ED7D31')
bars3 = ax.bar(x + width, few_shot_f1, width, label='Few-shot CoT F1', color='#A5A5A5')

ax.set_ylabel('F1-score', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Figure 3. Version 2 F1-score by Prompt Type', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure3_version2_f1.png', dpi=150)
plt.show()