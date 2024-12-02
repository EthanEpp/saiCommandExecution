import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
models = ['SOTA on SNIPS', 'In House Command Processor']
f1_scores = [98.30, 99.93]  # SOTA F1 score and your model F1 score
accuracy = [99.42, 99.75]  # SOTA accuracy and your model accuracy
plt.rcParams.update({'font.size': 16})  # Increase the default font size
# X positions for the bars
x = np.arange(len(models))

# Bar width
width = 0.35
# Adjusting the bar width and creating two separate plots for F1 Score and Intent Accuracy
# Adjusting the layout to add space between the title, labels, and making the colors consistent for the models
# Adjusting colors and background as requested: black background, lighter blue, and magenta instead of orange
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')

# X positions for the bars
x = np.arange(len(models))

# Adjusted bar width
width = 0.2

# Colors for models: SOTA in magenta, In House in lighter blue
colors_f1 = ['#F02360', '#4C23F0']  # Lighter blue for in-house model
colors_acc = ['#F02360', '#4C23F0']

# Setting background colors for axes and text
for ax in [ax1, ax2]:
    ax.set_facecolor('black')  # Set plot background color
    ax.tick_params(axis='x', colors='white')  # Set tick color for x-axis
    ax.tick_params(axis='y', colors='white')  # Set tick color for y-axis
    ax.yaxis.label.set_color('white')  # Set y-axis label color
    ax.xaxis.label.set_color('white')  # Set x-axis label color
    ax.title.set_color('white')  # Set title color

# Plot 1: F1 Score Comparison
ax1.bar(x, f1_scores, width, color=colors_f1)
ax1.set_title('F1 Score Comparison', fontsize=16, pad=20)
ax1.set_xlabel('Models', fontsize=14, labelpad=20)
ax1.set_ylabel('F1 Score (%)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylim([95, 100])

# Display the values on top of the bars
for i, v in enumerate(f1_scores):
    ax1.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, color='white')

# Plot 2: Intent Accuracy Comparison
ax2.bar(x, accuracy, width, color=colors_acc)
ax2.set_title('Intent Accuracy Comparison', fontsize=16, pad=20)
ax2.set_xlabel('Models', fontsize=14, labelpad=20)
ax2.set_ylabel('Accuracy (%)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=12)
ax2.set_ylim([95, 100])

# Display the values on top of the bars
for i, v in enumerate(accuracy):
    ax2.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, color='white')

plt.tight_layout()
plt.show()