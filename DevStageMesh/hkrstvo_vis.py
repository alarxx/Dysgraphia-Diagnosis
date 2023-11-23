import matplotlib.pyplot as plt

# Data for visualization
word = "lamoken"
accuracy = 66.0

# Creating the figure
plt.figure(figsize=(10, 6))

# Displaying the word
plt.text(0.5, 0.6, word, fontsize=40, ha='center')

# Displaying the accuracy below the word
plt.text(0.5, 0.4, f"{accuracy}%", fontsize=30, ha='center', color='green')

# Setting the limits and turning off the axis
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Title
plt.title('Word Visualization with Accuracy Percentage', fontsize=15)

# Displaying the plot
plt.show()
