from zoneinfo._common import load_data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import outlier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from fontTools.diff import summarize

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
features = ['CGPA', 'Internships', 'Projects', 'CodingTestScore']

X = df[features]  # independet variables
y = df[target]  # target / dependent

print(X.shape, y.shape)
print("=================================================================================")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
num_cols = df.select_dtypes('number').columns
cat_cols = df.select_dtypes('object').columns
print(num_cols)
print(cat_cols)

df['CGPA_Category'] = pd.cut(df["CGPA"],
                             bins=[0, 6, 8, 10], labels=['low', 'medium', 'high'], ordered=True)

print(df[['CGPA', 'CGPA_Category']].head())
print("=================================================================================")
# fig, axes = plt.subplots(2, 2, figsize=(14,10))
# sns.histplot(df["CGPA"],kde=True,ax=axes[0,0])
# sns.boxplot(x=df['CGPA'],ax=axes[0,1])
# sns.scatterplot(x='CGPA', y='PlacementStatus', data=df, ax=axes[1,0])
# sns.heatmap(df.corr(numeric_only=True), annot=True, ax=axes[1,1])
# plt.tight_layout()
# plt.show()
print("=================================================================================")
df['CGPA'].describe()
sns.histplot(df['CGPA'])
plt.show()

from scipy import stats

stats.ttest_ind(
    df[df.PlacementStatus == 'Yes']['CGPA'],
    df[df.PlacementStatus == 'No']['CGPA'], )
print("=================================================================================")
df['CGPA'].max() - df['CGPA'].min()

df['CGPA'].var()
df['CGPA'].std()
q1, q3 = df['CGPA'].quantile([0.25, 0.75])
iqr = q3 - q1
print("\nSummary Statistics:")
print(df['CGPA'].describe())
print("=================================================================================")
# df['branch'].value.count()
# sns.countplot(x=['barnch'], data=df, order=df['branch'].value_counts().index)
# df['branch'].values_count().plot.pie(autopct='%1.0f%%')
# print("=================================================================================")
# # Missing percentage
# missing_percent = (df.isnull().sum()/len(df))*100
# print(missing_percent)
#
# # Heatmap
# plt.figure(figsize=(10,5))
# sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
# plt.show()
#
# # Fill numeric missing values
# for col in df.select_dtypes(include='number').columns:
#     df[col].fillna(df[col].median(), inplace=True)
# print("=================================================================================")
# plt.figure(figsize=(6,5))
# sns.countplot(x='PlacementStatus', data=df)
# plt.title("Placement Status")
# plt.show()
# print("=================================================================================")
# cols=['CGPA','Attendance','AptitudeTestScore',
#       'CodingTestScore','SoftSkillsRating',
#       'MockInterviewScore']
#
# for col in cols:
#     if col in df.columns:
#         plt.figure(figsize=(6,4))
#         sns.histplot(df[col], kde=True)
#         plt.title(col)
#         plt.show()
# print("=================================================================================")
# cols=['CGPA','AptitudeTestScore',
#       'CodingTestScore',
#       'SoftSkillsRating',
#       'MockInterviewScore',
#       'Salary Package']
#
# for col in cols:
#     if col in df.columns:
#         plt.figure(figsize=(6,4))
#         sns.boxplot(x=df[col])
#         plt.title(col)
#         plt.show()
# print("=================================================================================")
# plt.figure(figsize=(12,8))
# sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm')
# plt.show()
# print("=================================================================================")
# sns.regplot(data=df,x='CGPA',y='Salary Package')
# plt.show()
# sns.regplot(data=df,x='AptitudeTestScore',y='CodingTestScore')
# plt.show()
print("=================================================================================")
import pandas as pd
from sklearn.model_selection import train_test_split


train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
from sklearn.preprocessing import StandardScaler

# Numerical columns
num_cols = [
    'CGPA',
    'Internships',
    'Projects',
    'Workshops'
]

# Create scaler
scaler = StandardScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(train_df[num_cols])

# Scale test data using the same scaler
X_test_scaled = scaler.transform(test_df[num_cols])

# Print learned statistics
print("Learned means:", scaler.mean_.round(2))
print("Learned std:", scaler.scale_.round(2))
print("=================================================================================")
print("This is onehot encoding")
def onehot():
    df = pd.read_csv('D:/ML/placement_predict_50k Dataset.csv')
    nominal_cols = ['Gender']

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['PlacementStatus'])
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    train_ohe = ohe.fit_transform(train_df[nominal_cols])
    test_ohe = ohe.transform(test_df[nominal_cols])
    feature_names = ohe.get_feature_names_out(nominal_cols)
    train_ohe_df = pd.DataFrame(train_ohe, columns=feature_names)
    print(train_ohe_df.head())
onehot()
print("=================================================================================")
print("This is Ordinal encoding")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
def ordinal():
    df = pd.read_csv('D:/ML/placement_predict_50k Dataset.csv')
    df['CGPA_Category'] = pd.cut(
        df["CGPA"],
        bins=[0, 6, 8, 10],
        labels=['low', 'medium', 'high'],
        ordered=True
    )
    ordinal_cols = ['CGPA_Category']
    categories_order = [['low', 'medium', 'high']]
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['PlacementStatus']
    )
    encoder = OrdinalEncoder(
        categories=categories_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    train_df_encoded[ordinal_cols] = encoder.fit_transform(train_df[ordinal_cols])
    test_df_encoded[ordinal_cols] = encoder.transform(test_df[ordinal_cols])
    print(train_df_encoded[ordinal_cols].head())

ordinal()
print("=================================================================================")
print("This is Label encoding for Branch")
import pandas as pd
def label_encode_branch():
    df = pd.read_csv('D:/ML/placement_predict_50k Dataset.csv')
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['PlacementStatus']
    )
    encoder = LabelEncoder()
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    # Label encoding the 'Stream' column (since 'Stream' represents the branch/department in this dataset)
    train_df_encoded['Stream'] = encoder.fit_transform(train_df['Stream'])
    test_df_encoded['Stream'] = encoder.transform(test_df['Stream'])
    print("Stream Classes mapped:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
    print(train_df_encoded['Stream'].head())
label_encode_branch()