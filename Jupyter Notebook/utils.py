import matplotlib.pyplot as plt
import seaborn as sns


def histograms_of_numerical_features(df, title: str):
    df.hist(bins=30, figsize=(12, 8))
    plt.suptitle(title)
    plt.show()


def value_counts_for_categorical_features(df):
    for col in df.select_dtypes(include='object').columns:
        print(f"\nValue counts for {col}:\n", df[col].value_counts().head(10))


def heatmap_correlation(df, title: str):
    numeric_cols = df.select_dtypes(include='number')
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()


def exclude_bots_from_df(df):
    # Define the nicknames to exclude
    exclude_nicknames = ['BetterBot', 'STEEBot', 'HastyBot']

    # Create a boolean mask
    mask = df['nickname'].isin(exclude_nicknames)

    # Use ~mask to get only the rows NOT in the list
    train_df_without_bots = df[~mask]

    return train_df_without_bots
