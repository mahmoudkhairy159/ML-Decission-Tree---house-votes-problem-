# ML Decision Tree - U.S. Congress Voting Record Problem

## Overview
This project involves building a **Decision Tree** model to predict the political party (Democrat or Republican) of members of the U.S. Congress based on their voting records from 1984. The dataset consists of voting results (Yes or No) on sixteen key issues for each of the 435 members of Congress.

The objective is to:
1. Train a decision tree using part of the dataset and evaluate its accuracy in predicting the political party.
2. Measure the impact of the training set size and random splits on both the accuracy and the size of the learned decision tree.
3. Handle missing values by imputing them with the majority vote for each issue.

## Dataset Description
The dataset contains:
- **16 features**: Voting results (Yes or No) on various issues.
- **1 label**: The political party of the congress member (Democrat or Republican).
- **435 instances**: Each instance represents the voting record of a member of Congress.

## Problem Statement
The goal is to train a Decision Tree model to predict the political party based on voting records. You'll explore the impact of different random splits of the data, as well as varying training set sizes, on the accuracy and size of the decision tree.

## Steps

### 1. Data Preparation
- **Handle Missing Values**: For each vote where a member of Congress did not participate, the missing value is replaced by the majority vote for that issue.
- **Data Splitting**: Randomly split the dataset into training and testing sets. The first experiment uses 25% of the data for training and the rest for testing. This process is repeated five times with different random splits to observe variations in tree size and accuracy.

### 2. Experiments
- **Initial Experiment**: Use 25% of the data for training and 75% for testing. Run the experiment five times with different random splits, noting the accuracy and tree size for each run.
- **Varying Training Set Size**: Measure the impact of training set size on accuracy and tree size. Perform experiments using different training set sizes (30%, 40%, 50%, 60%, and 70%), running each experiment with five different random splits.
  
### 3. Metrics
- **Accuracy**: Evaluate the accuracy of the decision tree on the test set.
- **Tree Size**: Measure the number of nodes in the learned decision tree.

### 4. Results
- For each training set size (30%, 40%, ..., 70%), report:
  - **Mean Accuracy**: The average accuracy over five random splits.
  - **Max/Min Accuracy**: The highest and lowest accuracy observed across the five splits.
  - **Mean Tree Size**: The average number of nodes in the decision tree.
  - **Max/Min Tree Size**: The largest and smallest tree sizes observed.
  
### 5. Plotting Results
Generate two plots to visualize the results:
1. **Accuracy vs. Training Set Size**: Plot the mean, maximum, and minimum accuracies for each training set size.
2. **Tree Size vs. Training Set Size**: Plot the mean, maximum, and minimum tree sizes for each training set size.

### 6. Tools & Libraries
You can use any machine learning library to build the Decision Tree. Popular libraries include:
- **Python**: `scikit-learn`
- **R**: `rpart`

For plotting, you can use:
- **Python**: `matplotlib` or `seaborn`
- **R**: `ggplot2`

### 7. Example Code (Python)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load data (replace with actual data loading process)
# X, y = load_data()

# Replace missing values with majority vote
# X = impute_missing_values(X)

# Train and test using different training sizes and random splits
training_sizes = [0.3, 0.4, 0.5, 0.6, 0.7]
mean_accuracies = []
tree_sizes = []

for size in training_sizes:
    accuracies = []
    nodes_count = []
    
    for _ in range(5):  # Repeat 5 times with random splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=np.random.randint(1000))
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        nodes_count.append(model.tree_.node_count)
    
    mean_accuracies.append(np.mean(accuracies))
    tree_sizes.append(np.mean(nodes_count))

# Plot results
plt.plot(training_sizes, mean_accuracies, label='Accuracy')
plt.plot(training_sizes, tree_sizes, label='Tree Size')
plt.legend()
plt.show()
