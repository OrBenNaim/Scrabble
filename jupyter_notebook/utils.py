import matplotlib.pyplot as plt
import seaborn as sns


def histograms_of_numerical_features(df, title: str):
    """
    Create histograms for all numerical features in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing numerical features to visualize
    title : str
        The title to display above the histograms

    Returns:
    --------
    None
        Displays the histograms using matplotlib
    """

    df.hist(bins=30, figsize=(12, 8))
    plt.suptitle(title)
    plt.show()


def value_counts_for_categorical_features(df):
    """
    Print value counts for all categorical features in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing categorical features to analyze

    Returns:
    --------
    None
        Prints the value counts for each categorical column
    """

    for col in df.select_dtypes(include='object').columns:
        print(f"\nValue counts for {col}:\n", df[col].value_counts().head(10))


def heatmap_correlation(df, title: str):
    """
   Create a heatmap showing correlations between all numerical features.

   Parameters:
   -----------
   df : pandas.DataFrame
       The DataFrame containing numerical features to calculate correlations
   title : str
       The title to display above the heatmap

   Returns:
   --------
   None
       Displays the correlation heatmap using seaborn
   """

    numeric_cols = df.select_dtypes(include='number')
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()


def exclude_bots_from_df(df):
    """
    Filter a DataFrame to exclude rows with bot nicknames.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing a 'nickname' column with both human and bot players

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with all rows where the nickname is not a bot

    Notes:
    ------
    Bots are identified by the following nicknames:
    - BetterBot
    - STEEBot
    - HastyBot
    """

    # Define the nicknames to exclude
    exclude_nicknames = ['BetterBot', 'STEEBot', 'HastyBot']

    # Create a boolean mask
    mask = df['nickname'].isin(exclude_nicknames)

    # Use ~mask to get only the rows NOT in the list
    train_df_without_bots = df[~mask]

    return train_df_without_bots
