import matplotlib.pyplot as plt
import numpy as np

datasets = ['Math', 'Commonsense', 'Multi-hop']
standard = [0.80, 0.74, 0.28]
zero_shot = [0.86, 0.46, 0.34]
few_shot = [0.96, 0.86, 0.42]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, standard, width, label='Standard', color='#4472C4')
bars2 = ax.bar(x, zero_shot, width, label='Zero-shot CoT', color='#ED7D31')
bars3 = ax.bar(x + width, few_shot, width, label='Few-shot CoT', color='#A5A5A5')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Figure 2. Version 2 Accuracy by Dataset and Prompt Type', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure2_version2_accuracy.png', dpi=150)
plt.show()