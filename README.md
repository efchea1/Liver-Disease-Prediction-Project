# Liver-Disease-Prediction-Project
This repository holds a solo liver disease prediction project.

# Project Overview

**Purpose**

This Python solo project focuses on predictive modeling for liver disease diagnosis using a synthetic dataset sourced from Kaggle. The primary aim is to leverage machine learning algorithms to identify patterns and factors influencing liver health, aiding in research insights and potential healthcare interventions.

### Dataset Description**

The dataset comprises 1,700 records and 11 features, encompassing **demographic, lifestyle,** and **health indicators**. The target variable is **"Diagnosis,"** a *binary indicator (0 or 1)* representing the presence or absence of liver disease. 

**The features include:**

* **Age:** Age of the individual (20-80 years).

* **Gender:** Male (0) or Female (1).

* **BMI (Body Mass Index):** 15-40.

* **Alcohol Consumption:** Weekly alcohol consumption (0-20 units).

* **Smoking:** Non-smoker (0) or smoker (1).

* **Genetic Risk:** Low (0), Medium (1), High (2).

* **Physical Activity:** Weekly hours of physical activity (0-10).

* **Diabetes:** No (0) or Yes (1).

* **Hypertension:** No (0) or Yes (1).

* **Liver Function Test:** 20-100.

This synthetic dataset was preprocessed to eliminate noise and irrelevant information, ensuring the focus remains on feature engineering and model development.

### Objectives

1. Conduct exploratory data analysis (EDA) to understand the dataset.

2. Preprocess the data by handling missing values and scaling features.

3. Apply machine learning algorithms, including Random Forest, Support Vector Machine (SVM), and Neural Network, to predict liver disease.

4. Compare model performance based on metrics such as accuracy, precision, recall, and F1-score.

5. Identify and save the best-performing model for deployment.

### Methodology

1. **EDA:** Explored data distribution, summary statistics, and potential correlations between features.

2. **Data Splitting:** Divided the dataset into **training (80%)** and **testing (20%)** subsets using *stratified sampling*.

3. **Feature Scaling:** Standardized features using StandardScaler for consistency across models.

4 **Model Training:**

* Random Forest Classifier

* Support Vector Machine (SVM) with RBF kernel

* Neural Network (MLPClassifier) with early stopping

5. **Model Evaluation:** Assessed performance using accuracy, confusion matrix, and classification reports.

6 **Model Comparison:** Determined the best model based on accuracy.

### Results and Findings

**Model Performance**

1. **Random Forest Classifier:**

* **Accuracy:** 89%

* **Precision:** 85% (class 0), 93% (class 1)

* **Recall:** 92% (class 0), 87% (class 1)

* **F1-Score:** 89% (class 0), 90% (class 1)

2. **SVM:**

* **Accuracy:** 85%

* **Precision:** 82% (class 0), 87% (class 1)

* **Recall:** 85% (class 0), 84% (class 1)

* **F1-Score:** 83% (class 0), 86% (class 1)

3. **Neural Network:**

* **Accuracy:** 82%

* **Precision:** 80% (class 0), 84% (class 1)

* **Recall:** 80% (class 0), 84% (class 1)

* **F1-Score:** 80% (class 0), 84% (class 1)

### Key Insights

1. Random Forest outperformed the other models with an accuracy of 89% and balanced precision, recall, and F1-score.

2. SVM demonstrated robust performance but slightly underperformed compared to Random Forest.

3. Neural Network showed promise but required additional fine-tuning to achieve comparable results.

### Best Model

The Random Forest Classifier was selected as the best-performing model based on its superior accuracy and classification metrics. This model has been saved for future deployment.

### Conclusion

This project highlights the application of machine learning techniques to predict liver disease effectively. The insights gained can inform healthcare interventions, supporting early detection and personalized treatment strategies. The Random Forest Classifier serves as a reliable model for this task, demonstrating high accuracy and robust performance metrics.

### Future Work

1. Explore additional feature engineering techniques to enhance model performance.

2. Experiment with other advanced machine learning algorithms.

3. Incorporate external datasets to validate the model’s generalizability.

4. Develop a user-friendly interface for deploying the model in real-world scenarios.

### Acknowledgments

* Dataset provided by Kaggle contributor Rabie El Kharoua.

* Machine learning techniques applied using Python libraries: ***pandas, numpy, scikit-learn.***

### License

This project is for educational purposes and also to give employers an understanding of what I have been doing in my free time. The dataset and code are freely available under the respective licenses outlined in the repository.
