import matplotlib.pyplot as plt
import numpy as np

datasets = ['Math', 'Commonsense', 'Multi-hop']
standard_steps = [4.78, 0.00, 0.90]
zero_shot_steps = [5.04, 0.84, 4.08]
few_shot_steps = [6.24, 0.00, 1.72]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, standard_steps, width, label='Standard Steps', color='#4472C4')
bars2 = ax.bar(x, zero_shot_steps, width, label='Zero-shot CoT Steps', color='#ED7D31')
bars3 = ax.bar(x + width, few_shot_steps, width, label='Few-shot CoT Steps', color='#A5A5A5')

ax.set_ylabel('Average Reasoning Steps', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_title('Figure 4. Version 2 Average Reasoning Steps by Prompt Type', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='upper right')
ax.set_ylim(0, 7)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_reasoning_steps.png', dpi=150)
plt.show()