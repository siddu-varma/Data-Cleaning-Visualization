import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
df = pd.read_csv("data/Titanic-Dataset.csv")

print("=" * 50)
print("TITANIC DATA CLEANING & VISUALIZATION PROJECT")
print("=" * 50)

# -----------------------------
# STEP 2: Dataset Overview
# -----------------------------
print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# -----------------------------
# STEP 3: Data Cleaning
# -----------------------------

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df = df.drop(columns=["Cabin"])

# Remove duplicate rows
df = df.drop_duplicates()

# -----------------------------
# STEP 4: Remove Outliers (Fare)
# -----------------------------
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# -----------------------------
# STEP 5: Visualizations
# -----------------------------

# Style
sns.set(style="whitegrid")

# 1 Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("images/age_distribution.png")
plt.close()

# 2 Gender Distribution
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Sex")
plt.title("Gender Distribution")
plt.savefig("images/gender_distribution.png")
plt.close()

# 3 Passenger Class
plt.figure(figsize=(6,6))
df["Pclass"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Passenger Class Distribution")
plt.savefig("images/passenger_class.png")
plt.close()

# 4 Survival Count
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.savefig("images/survival_count.png")
plt.close()

# 5 Fare Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Distribution")
plt.savefig("images/fare_boxplot.png")
plt.close()

# 6 Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.close()

print("\nAll graphs saved successfully in the images folder.")

# -----------------------------
# STEP 6: Insights
# -----------------------------
print("\nPROJECT INSIGHTS")
print("-" * 40)

print("Total Passengers:", len(df))
print("Average Age:", round(df["Age"].mean(),2))
print("Average Fare:", round(df["Fare"].mean(),2))
print("Male Passengers:", (df["Sex"]=="male").sum())
print("Female Passengers:", (df["Sex"]=="female").sum())
print("Passengers Survived:", (df["Survived"]==1).sum())
print("Passengers Not Survived:", (df["Survived"]==0).sum())

print("\nProject Completed Successfully!")