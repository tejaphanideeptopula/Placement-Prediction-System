import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('D:/ML/placement_predict_50k Dataset.csv')

print(df.head())
print("==============================")
print(df.info())
print("==============================")
print(df.describe())
print("==============================")
print(df.shape)
print("==============================")
print(df.isnull().sum())
print("==============================")
print(df.tail())
print("==============================")
print(df.duplicated().sum())
print("==============================")
print(df.nunique())
print("==============================")
print(df.columns.tolist())
# print("==============================")
#
# plt.figure(figsize=(8,5))
#
# sns.scatterplot(
#     data=df,
#     x='CGPA',
#     y='Salary Package'
# )
#
# plt.title('Salary Package vs CGPA')
# plt.xlabel('CGPA')
# plt.ylabel('Salary Package')
# plt.show()
print("=================================================================================")

target = 'PlacementStatus'
features = ['CGPA', 'Internships','Projects','CodingTestScore']

X = df[features]     # independet variables
y = df[target]      # target / dependent

print(X.shape , y.shape)
print("=================================================================================")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
print("=================================================================================")
# Mean
mean = df['CGPA'].mean()
print("Mean:", mean)

# Median
median = df['CGPA'].median()
print("Median:", median)

# Mode
mode = df['CGPA'].mode()
print("Mode:", mode.tolist())  # Handles one or more modes

print("=================================================================================")
num_cols  = df.select_dtypes('number').columns
cat_cols = df.select_dtypes('object').columns
print(num_cols)
print(cat_cols)

df['CGPA_Category'] = pd.cut(df["CGPA"],
bins=[0,6,8,10], labels=['low' , 'medium', 'high'],
ordered=True)

print(df[['CGPA', 'CGPA_Category']].head())
print("=================================================================================")
fig, axes = plt.subplots(2, 2, figsize=(14,10))
sns.histplot(df["CGPA"],kde=True,ax=axes[0,0])
sns.boxplot(x=df['CGPA'],ax=axes[0,1])
sns.scatterplot(x='CGPA', y='PlacementStatus', data=df, ax=axes[1,0])
sns.heatmap(df.corr(numeric_only=True), annot=True, ax=axes[1,1])
plt.tight_layout()
plt.show()