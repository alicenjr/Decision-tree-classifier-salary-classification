# 🌳 Decision Tree Classifier - Salary Classification

A machine learning project that uses Decision Tree algorithm to predict salary categories based on various demographic and professional features. This project demonstrates classification techniques and model evaluation.

## 📋 Overview

This project implements a Decision Tree Classifier to predict whether an individual's salary falls into different categories (e.g., above or below a threshold). Decision trees are powerful, interpretable machine learning models that make predictions through a series of decision rules.

## ✨ Features

- **Decision Tree Classification**: Implements tree-based classification algorithm
- **Feature Analysis**: Identifies most important features for salary prediction
- **Model Visualization**: Visual representation of the decision tree structure
- **Performance Metrics**: Comprehensive model evaluation
- **Interpretability**: Easy to understand decision rules
- **Jupyter Notebook**: Interactive analysis and experimentation

## 🎯 What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that:
- Makes decisions through a tree-like structure
- Splits data based on feature values
- Creates rules for classification
- Is highly interpretable and visual
- Handles both numerical and categorical data

**Key Advantages:**
- Easy to understand and interpret
- Requires little data preprocessing
- Can handle non-linear relationships
- Provides feature importance
- Works with both classification and regression

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Machine learning libraries

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alicenjr/Decision-tree-classifier-salary-classification.git
cd Decision-tree-classifier-salary-classification
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Open the Jupyter notebook:
```bash
jupyter notebook decision_tree.ipynb
```

## 📊 Dataset

The `salaries.csv` dataset contains features such as:
- **Age**: Individual's age
- **Education Level**: Highest education attained
- **Years of Experience**: Work experience in years
- **Job Title/Role**: Current position
- **Hours per Week**: Working hours
- **Occupation**: Type of work
- **Salary**: Target variable (salary category)

## 🔧 Implementation Steps

### 1. Data Loading & Exploration
```python
import pandas as pd
df = pd.read_csv('salaries.csv')
df.head()
df.info()
df.describe()
```

### 2. Data Preprocessing
- Handle missing values
- Encode categorical variables (Label Encoding / One-Hot Encoding)
- Feature scaling (if needed)
- Train-test split

### 3. Model Training
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',  # or 'entropy'
    max_depth=10,
    min_samples_split=20,
    random_state=42
)

model.fit(X_train, y_train)
```

### 4. Model Evaluation
- Accuracy score
- Confusion matrix
- Classification report (Precision, Recall, F1-Score)
- ROC-AUC curve

### 5. Tree Visualization
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Low', 'High'])
plt.show()
```

## 📈 Model Performance

Typical results for decision tree classifiers:
- **Accuracy**: 75-85% on test data
- **Interpretability**: High - can visualize decision rules
- **Training Time**: Fast
- **Prediction Time**: Very fast

## 💡 Key Concepts

### Splitting Criteria

**Gini Impurity:**
```
Gini = 1 - Σ(p_i)²
```
Measures the probability of incorrect classification.

**Entropy (Information Gain):**
```
Entropy = -Σ(p_i × log₂(p_i))
```
Measures the disorder or impurity in the data.

### Hyperparameters

- `max_depth`: Maximum depth of the tree
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node
- `criterion`: Splitting criterion ('gini' or 'entropy')

## 🔍 Feature Importance

Decision trees automatically provide feature importance scores:

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
```

## 📁 Project Structure

```
├── decision_tree.ipynb    # Main Jupyter notebook with implementation
├── salaries.csv          # Dataset with salary information
└── README.md            # Project documentation
```

## 🛠️ Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning library (Decision Tree, metrics)
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization
- **Jupyter Notebook**: Interactive development

## 🎯 Use Cases

Decision Tree Classification is widely used for:
- Credit risk assessment
- Medical diagnosis
- Customer churn prediction
- Employee attrition prediction
- Loan approval systems
- Fraud detection

## 📊 Evaluation Metrics

```
Confusion Matrix:
                 Predicted
               Low    High
Actual  Low    TN     FP
        High   FN     TP

Accuracy = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

## 🚀 Avoiding Overfitting

Decision trees can easily overfit. To prevent this:
- Set `max_depth` limit
- Increase `min_samples_split`
- Increase `min_samples_leaf`
- Use pruning techniques
- Employ Random Forest (ensemble method)

## 🎓 Learning Outcomes

This project demonstrates:
- Decision Tree algorithm implementation
- Classification problem-solving
- Feature engineering and selection
- Model evaluation techniques
- Hyperparameter tuning
- Interpretable machine learning

## 🤝 Contributing

Contributions are welcome! You can:
- Improve preprocessing pipeline
- Add cross-validation
- Implement pruning techniques
- Try ensemble methods (Random Forest, Gradient Boosting)
- Add more visualizations
- Optimize hyperparameters

## 📝 License

This project is open-source and available for educational and commercial use.

## 👨‍💻 Author

**alicenjr** - [GitHub Profile](https://github.com/alicenjr)

---

⭐ Star this repo if you find it useful!
