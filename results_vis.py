import matplotlib.pyplot as plt

# Data for the models and their accuracies
models = ['AdaBoost', 'SVM', 'RF', 'VGG16']
accuracies = [79.5, 78.8, 77.6, 79.17]

# Creating the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])

# Adding the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

# Adding labels and title
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Machine Learning Model Accuracies')
plt.ylim(0, 100)  # Setting the y-axis limit to show percentages clearly

# Displaying the plot
plt.show()
