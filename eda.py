import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# 1. Summary statistics
print("Summary statistics:")
print(df.describe(include='all'))
print("\nMissing values:")
print(df.isnull().sum())

# 2. Histograms for numeric features
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
sns.set(style="whitegrid")

for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Boxplots for numeric features
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.show()

# 3. Correlation matrix and heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 4. Patterns and trends

# Survival rate by gender
print("\nSurvival rate by gender:")
print(df.groupby('Sex')['Survived'].mean())

# Survival rate by class
print("\nSurvival rate by class:")
print(df.groupby('Pclass')['Survived'].mean())

# 5. Feature-level inferences using visualizations

# Age vs Survival (KDE plot)
plt.figure(figsize=(8, 4))
sns.kdeplot(df[df['Survived'] == 1]['Age'].dropna(), label='Survived', shade=True)
sns.kdeplot(df[df['Survived'] == 0]['Age'].dropna(), label='Did Not Survive', shade=True)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Fare vs Survival (Boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()
