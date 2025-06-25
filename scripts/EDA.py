import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("plots", exist_ok=True)

def run_eda(path="data/raw/xente_dataset.csv"):
    print("🔍 Reading data...")
    df = pd.read_csv(path)

    print("\n Shape:", df.shape)
    print("\n Data Types:\n", df.dtypes)
    print("\n Info:")
    print(df.info())
    print("\n Summary Statistics:\n", df.describe(include='all').T)

    # Plot numerical feature distributions
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols].hist(figsize=(15, 10), bins=30)
    plt.tight_layout()
    plt.savefig("plots/numerical_distributions.png")
    plt.close()

    # Categorical feature distributions
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10])
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"plots/categorical_{col}.png")
        plt.close()

    # Correlation heatmap
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.close()

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\n Missing Values:\n", missing)

    # Boxplots for outliers
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot for {col}')
        plt.tight_layout()
        plt.savefig(f"plots/boxplot_{col}.png")
        plt.close()

    print(" EDA completed. Plots saved to 'plots/' folder.")

if __name__ == "__main__":
    run_eda()