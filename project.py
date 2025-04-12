import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


# EDA
# Load the dataset
df=pd.read_csv(r'C:\Users\Piyush\Downloads\Census income dataset.csv')

# Display first few rows
print(df.head())
print("\n")

# Show column names and data types
print(df.info())
print("\n")

# Check for missing values
print("\nMissing Values:\n",df.isnull().sum())

print("\nMissing Values:",df.isnull().sum().sum())

# Summary of numerical columns
print("\n",df.describe())

# Summary of categorical columns
print("\n",df.describe(include="object"))



# Data Visualiation
# Histograms for numerical data
sns.set(style="whitegrid")
numerical_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
df[numerical_cols].hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

# Correlation heatmap for numerical columns
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()



# Hypothesis Testing
# T-test
# Clean class column (strip whitespace and periods)
df['class'] = df['class'].str.strip().str.replace('.', '', regex=False)

# Create groups
low_income = df[df["class"] == "<=50K"]["hours-per-week"]
high_income = df[df["class"] == ">50K"]["hours-per-week"]

# Perform independent t-test
t_stat, p_value = ttest_ind(high_income, low_income, equal_var=False)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in average hours worked per week.")
else:
    print("Fail to reject the null hypothesis: No significant difference in average hours worked per week.")

# chi-square test
contingency_table = pd.crosstab(df['marital-status'], df['class'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Statistic:", chi2)
print("P-value:", p)




# Effect of Education on Income
# Examine how income varies with different levels of education.
education_income = df.groupby(['education', 'class']).size().unstack().fillna(0)
education_income.plot(kind='bar', stacked=True, figsize=(12,6), colormap='viridis')
plt.title('Income Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()




# Capital Gains/Losses Impact
# Evaluate how capital gains and losses influence income classification.

sns.boxplot(data=df, x='class', y='capital-gain')
plt.title('Capital Gain by Income Class')
plt.xlabel('Income Class')
plt.ylabel('Capital Gain')
plt.show()

sns.boxplot(data=df, x='class', y='capital-loss')
plt.title('Capital Loss by Income Class')
plt.xlabel('Income Class')
plt.ylabel('Capital Loss')
plt.show()




#Outlier Detection using IOR Method
#Selection numerical columns

numerical_columns=df.select_dtypes(include=['float64','int64']).columns

Q1=df[numerical_columns].quantile(0.25)
Q3=df[numerical_columns].quantile(0.75)
IQR=Q3-Q1

#Define upper and lower bound
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

outliers_iqr=((df[numerical_columns]<lower_bound)|(df[numerical_columns]>upper_bound))
print(outliers_iqr)
df[numerical_columns].boxplot(rot=45)
plt.title("Box Plot For Outlier Detection")
plt.show()
