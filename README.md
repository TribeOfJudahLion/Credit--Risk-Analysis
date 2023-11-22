<br/>
<p align="center">
  <h3 align="center">Unveil Financial Insight: Master Credit Risk Analysis</h3>

  <p align="center">
    Navigate the credit maze with confidence — Decode risk, safeguard your portfolio.
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Statement:

A financial institution is facing challenges in evaluating the credit risk associated with loan applicants. Traditional methods have proven to be time-consuming and not as accurate, resulting in a higher rate of loan defaults. The institution is in need of a more efficient, automated, and data-driven approach to assess and predict the creditworthiness of loan applicants.

### Solution:

To address this problem, a machine learning model was developed to predict the likelihood of loan default based on historical loan application data. The model uses various features related to the applicants' financial background and loan details.

#### Data Processing and Feature Engineering:

1. **Data Loading**: The dataset `credit_risk_dataset.csv`, which includes historical loan application records with various attributes and the loan status (defaulted or not), is loaded into a Pandas DataFrame.

2. **Feature Selection**: Numerical and categorical features are identified. Numerical features include income, loan amount, etc., while categorical features comprise loan purpose, home ownership status, etc.

3. **Preprocessing**:
   - Numerical features are processed with median imputation for missing values and scaling to standardize the range of data.
   - Categorical features undergo imputation for missing values using the most frequent value and are encoded using one-hot encoding to convert them into a format that can be provided to machine learning algorithms.

#### Model Training and Evaluation:

1. **Model Choice**: A RandomForestClassifier is selected due to its robustness and ability to handle non-linear relationships between features.

2. **Training**: The model is trained using the preprocessed features to learn the patterns associated with loan defaults.

3. **Cross-Validation**: A 5-fold cross-validation is performed to evaluate the model's performance, particularly using the ROC AUC metric to account for both the true positive and false positive rates.

4. **Prediction**: The model is used to make predictions on a separate test dataset that the model has not seen during training to evaluate its real-world performance.

#### Performance Metrics:

1. **ROC AUC Score**: An average 5-fold ROC AUC score of 0.927 indicates that the model has a high discriminative ability to differentiate between the loan statuses.

2. **Classification Report**: Precision, recall, and F1-scores for each class (defaulted or not) are calculated, showing high precision (0.93) and a good recall (0.99) for the negative class, and even higher precision (0.97) but lower recall (0.72) for the positive class.

3. **Confusion Matrix**: It reveals how the model's predictions match up against the actual loan statuses, providing a clear picture of true positives, true negatives, false positives, and false negatives.

#### Visualizations:

1. **Feature Importance**: A bar chart is generated to visualize the importance of each feature used in the model. This helps in understanding which features have the most influence on the model's decisions.

2. **ROC Curve**: A plot showing the true positive rate against the false positive rate at various threshold levels, with the AUC score indicating the quality of the model's predictions.

3. **Confusion Matrix Visualization**: A heatmap is used to visualize the confusion matrix, providing an intuitive display of the model's performance.

#### Model Deployment:

The trained model is saved to a file (`credit_risk_model.pkl`) using `joblib`, which allows it to be deployed in a production environment where it can be used to make predictions on new loan applications.

### Conclusion:

By utilizing machine learning, the financial institution can now rapidly and accurately assess the credit risk of loan applicants. The model's high predictive accuracy can help reduce the number of loan defaults, thereby saving time and resources while also allowing the institution to extend credit to more deserving applicants with confidence.

Here's a detailed breakdown of each part:

### Cell 1: Importing Libraries
- **Functionality**: This cell imports necessary libraries for data processing, machine learning modeling, visualization, and logging. Libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn are essential for handling data frames, mathematical operations, plotting graphs, and machine learning tasks.

### Cell 2: Setting up Logging
- **Logic**: This cell sets up the logging system to track the progress and status of the program execution. It's crucial for debugging and monitoring the model's performance during training and evaluation.

### Cell 3: Loading the Dataset
- **Functionality**: The cell loads a dataset from a CSV file. The dataset likely contains information relevant to credit risk assessment.
- **Context**: The dataset is assumed to be named 'credit_risk_dataset.csv' and should be located in the same directory as the notebook or at a specified path.

### Cell 4: Identifying Categorical and Numerical Columns
- **Logic**: This cell identifies and separates the categorical and numerical columns from the dataset. This distinction is vital for appropriate preprocessing steps in the machine learning pipeline.

### Cell 5: Data Preprocessing
- **Functionality**: The cell prepares the dataset for modeling by separating the features (`X`) and the target variable (`y`). The target variable is typically the outcome we want to predict, in this case, likely the credit risk status.

### Cell 6: Splitting the Dataset
- **Functionality**: Here, the dataset is split into training and testing sets. This is a standard practice in machine learning to evaluate the model on unseen data.

### Cell 7: Defining the Preprocessing for Numerical and Categorical Features
- **Logic**: This cell creates a preprocessing pipeline for both numerical and categorical features using Scikit-Learn's `ColumnTransformer`. This step is essential for handling different data types appropriately.

### Cell 8: Creating and Training the Pipeline
- **Functionality**: A machine learning pipeline is created and trained. This pipeline includes the preprocessing steps and a Random Forest classifier for the prediction task.

### Cell 9: Model Evaluation with Cross-Validation
- **Functionality**: The model's performance is evaluated using cross-validation, specifically measuring the ROC AUC score. This provides an insight into the model's generalization capability.

### Cell 10: Making Predictions and Evaluating the Model
- **Functionality**: The cell uses the trained model to make predictions on the test set and evaluates these predictions using different metrics like the classification report and ROC AUC score.

### Cell 11: Confusion Matrix Visualization
- **Functionality**: A confusion matrix is plotted to visualize the performance of the model in terms of true positives, true negatives, false positives, and false negatives.

### Cell 12: ROC Curve Visualization
- **Functionality**: This cell plots the ROC curve, a graphical representation that illustrates the diagnostic ability of a binary classifier.

### Cell 13: Feature Importance Visualization
- **Functionality**: The cell visualizes the importance of each feature used by the model. This is helpful to understand which features are most influential in predicting the target variable.

### Cell 14: Saving the Model
- **Functionality**: Finally, the trained model pipeline is saved to a file using `joblib`. This allows the model to be reused later without retraining.

Throughout the notebook, each cell is dedicated to a specific task or step in the machine learning workflow, from data loading and preprocessing to model training, evaluation, and saving. The notebook's structure makes it easy to understand and follow the sequence of operations necessary for building and evaluating a machine learning model for credit risk assessment.

The output of the machine learning model training and evaluation process consists of three visualizations and several performance metrics. Let's dissect them one by one.

### Feature Importances Visualization (output1.png)

This bar chart shows the relative importance of each feature used by the Random Forest Classifier to make predictions. The most important features contribute more to the decision-making process of the model. Here are some insights:

- **loan_percent_income** is the most significant feature, suggesting that the percentage of income required to service the loan is a strong predictor of credit risk.
- **person_income**, **loan_int_rate**, and **loan_amnt** also appear to be important features, indicating that the borrower's income, the interest rate of the loan, and the amount of the loan are critical in assessing credit risk.
- The importance of features drops off as we move to the right of the graph, showing that many features contribute only marginally to the predictions.

### Receiver Operating Characteristic (ROC) Curve (output2.png)

The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. Here's what the curve indicates:

- **Area Under the Curve (AUC)**: The area under the ROC curve is 0.93, which is close to 1. This suggests that the model has a high degree of separability and can distinguish between the classes well.
- **Curve Shape**: The curve bows towards the upper-left corner of the graph, which indicates good performance. The closer the curve comes to the top left corner, the better the model's ability to maximize true positives while minimizing false positives.

### Confusion Matrix (output3.png)

A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. The matrix includes:

- **True Positives (TP)**: Lower right cell (1557) — The model correctly predicted the positive class.
- **True Negatives (TN)**: Upper left cell (7567) — The model correctly predicted the negative class.
- **False Positives (FP)**: Upper right cell (46) — The model incorrectly predicted the positive class.
- **False Negatives (FN)**: Lower left cell (605) — The model incorrectly predicted the negative class.

### Model Performance Metrics

The log outputs provide various performance metrics:

- **Average 5-Fold ROC AUC**: 0.9273912789288048 — The average ROC AUC score across a 5-fold cross-validation is approximately 0.927, which indicates strong predictive power.
- **Precision**: Precision is the ratio of correctly predicted positive observations to the total predicted positives. High precision for class 1 (0.97) indicates the model has a low false positive rate for the positive class.
- **Recall**: Recall is the ratio of correctly predicted positive observations to all observations in the actual class. The recall for class 1 is 0.72, meaning that the model correctly identifies 72% of the actual positive cases.
- **F1-Score**: The F1 score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. An F1 score of 0.83 for class 1 is quite robust, indicating a good balance between precision and recall.
- **Accuracy**: The overall accuracy of the model is 0.93, showing that the model correctly predicts the credit risk status for 93% of the loans in the test set.
- **ROC AUC Score**: 0.9334778756771398 — The ROC AUC score for the test set predictions is approximately 0.933, which is consistent with the cross-validation score, underscoring the model's reliability.

In summary, the visualizations and metrics suggest that the model is performing very well, with strong predictive capabilities for identifying credit risk. The ROC curve and confusion matrix provide a comprehensive picture of the model's performance across different thresholds, balancing the true positive rate against the false positives. The precision, recall, and F1-scores offer additional detail about the model's accuracy and robustness for each class.

## Built With

This project utilizes a variety of tools and libraries for machine learning, data preprocessing, visualization, and logging. Below is a comprehensive list of components:

- **[Pandas](https://pandas.pydata.org/)** - A powerful data analysis and manipulation library for Python, used for reading data files and handling dataframes.
- **[NumPy](https://numpy.org/)** - Fundamental package for scientific computing with Python, providing support for array objects and mathematical operations.
- **[Matplotlib](https://matplotlib.org/)** - A plotting library for the Python programming language and its numerical mathematics extension NumPy, used for creating static, interactive, and animated visualizations.
- **[Seaborn](https://seaborn.pydata.org/)** - A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- **[Joblib](https://joblib.readthedocs.io/en/latest/)** - A set of tools to provide lightweight pipelining in Python, particularly used here for saving and loading the machine learning model.
- **[Scikit-Learn](https://scikit-learn.org/stable/)** - Machine learning library for Python, used for modeling and evaluating the machine learning pipeline, including:
  - **Model Selection**:
    - `train_test_split` - Splits arrays or matrices into random train and test subsets.
    - `cross_val_score` - Evaluates a score by cross-validation.
  - **Preprocessing**:
    - `StandardScaler` - Standardizes features by removing the mean and scaling to unit variance.
    - `OneHotEncoder` - Encodes categorical features as a one-hot numeric array.
    - `SimpleImputer` - Imputation transformer for completing missing values.
  - **Pipeline**:
    - `ColumnTransformer` - Applies transformers to columns of an array or pandas DataFrame.
    - `Pipeline` - Sequentially applies a list of transforms and a final estimator.
  - **Ensemble Methods**:
    - `RandomForestClassifier` - A meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control overfitting.
  - **Metrics**:
    - `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `auc` - Various metrics used to evaluate the performance of the machine learning model.
- **[Logging](https://docs.python.org/3/library/logging.html)** - Module in Python’s Standard Library for tracking events that happen when some software runs.

## Roadmap

See the [open issues](https://github.com//Credit--Risk-Analysis/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com//Credit--Risk-Analysis/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com//Credit--Risk-Analysis/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com//Credit--Risk-Analysis/blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion/) - **

## Acknowledgements

* []()
* []()
* []()
