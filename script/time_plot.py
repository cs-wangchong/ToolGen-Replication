import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)

# Data
categories_A = [r'\textsc{Vanilla}-\textit{gpt}', 'BaseTFive', 'BaseGen', 'BaseLlama']
categories_B = ['AppGPT', 'AppTFive', 'AppGen', 'AppLlama']
A = [0.26, 0.44, 0.51, 1.37]
B = [0.63, 0.84, 0.84, 2.34]

# Positions of the bars on the x-axis
bar_width = 0.35
x = np.arange(len(categories_A))

# Create the plot
fig, ax = plt.subplots()

# Plot the bars
bars1 = ax.bar(x - bar_width/2, A, bar_width, label='Group A')
bars2 = ax.bar(x + bar_width/2, B, bar_width, label='Group B')

xticks_A = []
xticks_B = []

# Adding the text on top of the bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    xticks_A.append(bar.get_x() + bar.get_width()/2)
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    xticks_B.append(bar.get_x() + bar.get_width()/2)

# Add labels, title, and legend
ax.set_ylabel(r'\textsc{Time (seconds)}')

# Set the custom x-tick labels
ax.set_xticks(xticks_A + xticks_B)
ax.set_xticklabels(categories_A + categories_B)


# # Add legend
# ax.legend()

dpi = 800
# plt.tight_layout()
plt.savefig('time-plot.png', dpi=dpi)
