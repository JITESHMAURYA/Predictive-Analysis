SAMPLING TECHNIQUES ON IMBALANCED CREDIT CARD DATASET
OBJECTIVE

The objective of this assignment is to analyze the impact of different sampling techniques on the performance of various machine learning models when dealing with a highly imbalanced dataset.
Imbalanced data is a common real-world problem, especially in fraud detection, where minority classes are often under-represented.

DATASET DESCRIPTION

The dataset used in this assignment is a credit card transaction dataset, where:

Class 0 represents normal (non-fraudulent) transactions

Class 1 represents fraudulent transactions

The dataset is highly imbalanced, with fraudulent transactions forming a very small portion of the data.
Such imbalance can severely affect machine learning model performance by biasing predictions toward the majority class.

Dataset source:
Creditcard_data.csv

METHODOLOGY
Handling Imbalanced Data

To address the imbalance problem, five different sampling techniques were applied directly to the original dataset:

Sampling Code	Technique	Description
Sampling1	Random Under Sampling	Reduces majority class samples
Sampling2	Random Over Sampling	Duplicates minority class samples
Sampling3	SMOTE	Generates synthetic minority samples
Sampling4	NearMiss	Selects closest majority samples
Sampling5	SMOTEENN	Hybrid of over & under sampling

Each technique aims to balance the dataset in a different way, allowing us to study their impact on model performance.

Machine Learning Models Used

Five commonly used machine learning models were trained on each sampled dataset:

Model Code	Algorithm
M1	Logistic Regression
M2	Decision Tree
M3	Random Forest
M4	K-Nearest Neighbors
M5	Support Vector Machine

For each sampling technique:

The dataset was split into 70% training and 30% testing

Model performance was evaluated using accuracy

RESULT TABLE

The table below shows the accuracy (%) obtained by each model under different sampling techniques:

Model	UnderSampling	OverSampling	SMOTE	NearMiss	SMOTEENN
Logistic Regression	50.10	52.24	63.18	69.23	70.12
Decision Tree	59.25	65.27	68.72	28.36	30.25
Random Forest	90.45	72.41	32.17	42.58	41.85
KNN	78.25	56.24	47.23	33.44	40.12
SVM	81.25	12.85	57.36	32.25	52.74

Note: Accuracy values may slightly vary when re-executed due to randomness in sampling.

RESULT GRAPH EXPLAINATION
🔹 1. Bar Graph (Models vs Sampling Techniques)

Displays accuracy comparison across all models and sampling techniques

Helps identify which sampling technique performs best for each model

Observation:

Random Forest shows the highest accuracy with Under Sampling

Logistic Regression benefits more from SMOTEENN

🔹 2. Line Graph (Accuracy Trends)

Shows accuracy variation of each model across sampling methods

Useful for understanding performance consistency

Observation:

No single sampling technique works best for all models

Model sensitivity to sampling varies significantly

🔹 3. Average Accuracy per Sampling Technique

Computes mean accuracy across all models for each sampling method

Observation:

SMOTE and SMOTEENN provide balanced performance

NearMiss often degrades performance due to excessive data removal

🔹 4. Heatmap-Style Visualization

Provides a quick visual overview of strong and weak model-sampling combinations

Darker regions indicate higher accuracy

KEY FINDINGS

Sampling techniques significantly influence model performance

Ensemble models (Random Forest) perform better with reduced noise

Linear models benefit from synthetic data generation

Aggressive under-sampling may lead to information loss

There is no universally best sampling technique

CONCLUSION

This study demonstrates that handling class imbalance is crucial for reliable machine learning performance.
The choice of sampling technique should depend on the model type and dataset characteristics.
Careful evaluation using multiple sampling strategies leads to better and more robust predictive models.
