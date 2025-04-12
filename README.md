📊 Overview
This project analyzes the Census Income Dataset to predict whether an individual earns more than $50K annually. The dataset is based on U.S. Census data and includes demographic, educational, and occupational attributes. The project explores various data science techniques including:

Exploratory Data Analysis (EDA)

Hypothesis Testing

Classification Modeling

Clustering

Feature Importance Analysis

Bias and Fairness Evaluation

📁 Dataset Description
The dataset is originally sourced from the UCI Machine Learning Repository and preprocessed into Excel format for ease of use.

Key Features:
Age: Age of the individual.

Workclass: Employment status.

Education: Highest level of education achieved.

Occupation: Job type.

Race & Sex: Demographic attributes.

Capital Gain/Loss: Investment income/loss.

Hours per week: Hours worked per week.

Income: Target variable (<=50K or >50K).

🧪 Project Objectives
Data Cleaning & Preprocessing

Handling missing values

Encoding categorical variables

Exploratory Data Analysis (EDA)

Univariate & bivariate visualizations

Distribution analysis

Hypothesis Testing

Checking correlations between variables and income

Validating statistical assumptions

Model Building

Logistic Regression, Random Forest, XGBoost, etc.

Performance metrics: Accuracy, Precision, Recall, F1-Score

Clustering

K-Means clustering to group individuals based on socioeconomic features

Feature Importance

Using models and statistical tools to identify key predictors of income

Bias & Fairness Analysis

Evaluating model bias across gender and race

Identifying potential discrimination

🛠 Tools and Technologies
Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)

Excel (Data formatting and report compilation)

Jupyter Notebook

SciPy/Statsmodels (for statistical testing)

📈 Results Summary
Best-performing model: Random Forest with ~85% accuracy

Top features: Education, Occupation, Capital Gain, Hours per week

Detected moderate bias in predictions based on Sex and Race

Clustering revealed three distinct socioeconomic profiles

