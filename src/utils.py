import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import TURNS_FILE_PATH, TRAIN_FILE_PATH, BOTS_NICKNAMES, HARD_LETTERS, SCRABBLE_LETTER_VALUES


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

    # Create a boolean mask
    mask = df['nickname'].isin(BOTS_NICKNAMES)

    # Use ~mask to get only the rows NOT in the list
    train_df_without_bots = df[~mask].copy()  # Return a copy to avoid SettingWithCopyWarning

    return train_df_without_bots


def calculate_base_word_points(move: str) -> int:
    """
    Given a move string from a Scrabble turn, calculate the base tile score
    ignoring board bonuses and existing tiles.
    """
    if not isinstance(move, str):
        return 0

    base_points = 0
    for char in move:
        if char == '.' or not char.isalpha():
            continue  # ignore existing letters on the board and non-letters

        if char.islower():
            continue  # lowercase = blank tile used → 0 points

        base_points += SCRABBLE_LETTER_VALUES.get(char.upper(), 0)

    return base_points


def create_new_features(df: pd.DataFrame) -> None:
    """
    Enhances the given Scrabble turns dataframe with additional features for analysis.

    This function creates several game-specific features to help analyze player performance:
    - word_length: Number of letters in each played word
    - is_bingo: Boolean indicating if the word used 7+ letters (a "bingo" in Scrabble)
    - uses_hard_letters: Boolean indicating if the move used high-value rare tiles (J,K,Q,X,Z)
    - is_negative_turn: Boolean indicating if the turn resulted in negative points
    - is_pass: Boolean indicating if the player passed their turn
    - is_exchange: Boolean indicating if the player exchanged tiles
    - base_word_points: The raw point value of the letters used (without board bonuses)
    - extra_points: Additional points earned through bonus squares or connecting words

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe of Scrabble game turns with at minimum these columns:
        'move', 'points', 'turn_type'

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with added feature columns

    Notes:
    ------
    - Assumes HARD_LETTERS are defined globally (typically J, K, Q, X, Z)
    - Requires the calculate_base_word_points function to be defined
    """

    # Helper column: word length
    df.loc[:, 'word_length'] = df['move'].apply(
        lambda x: len(x) if isinstance(x, str) else 0)

    # Bingo: word length >= 7
    df.loc[:, 'is_bingo'] = df['word_length'] >= 7

    # Check if move uses any hard letters
    df.loc[:, 'uses_hard_letters'] = df['move'].apply(
        lambda x: any(letter in str(x).upper() for letter in HARD_LETTERS) if isinstance(x, str) else False
    )

    # Negative points
    df.loc[:, 'is_negative_turn'] = df['points'] < 0

    # Pass detection
    df.loc[:, 'is_pass'] = df['turn_type'].str.lower() == 'pass'

    # Exchange detection
    df.loc[:, 'is_exchange'] = df['turn_type'].str.lower() == 'exchange'

    # Adds a new 'extra_points' column to the DataFrame representing how many
    # points above base tile value a player earned on a turn — from bonuses, multipliers, or word connections.
    df.loc[:, 'base_word_points'] = df['move'].apply(calculate_base_word_points)
    df.loc[:, 'extra_points'] = df['points'] - df['base_word_points']

def create_training_examples() -> pd.DataFrame:
    """
    Creates dataset to predict user ratings based on game behavior.
    Returns:
            pd.DataFrame: A DataFrame where each row represents a game, with features and the target rating.
    """

    # === Step 0: Load Data ===
    turns_df = pd.read_csv(TURNS_FILE_PATH)
    train_df = pd.read_csv(TRAIN_FILE_PATH)

    # === Step 1: Remove bots ===
    train_df_without_bots = exclude_bots_from_df(turns_df)

    # Keep only rows in turns_df where 'game_id' exists in train_df_without_bots
    # (i.e., filter turns_df by matching game_ids)
    turns_df = turns_df[turns_df['game_id'].isin(train_df_without_bots['game_id'])]

    # Remove bots
    turns_df_without_bots = exclude_bots_from_df(turns_df)

    # === Step 2: Creates new features for turns_df_without_bots ===
    create_new_features(df=turns_df_without_bots)

    # === Step 3: # Aggregated features per player per game ===
    features = turns_df_without_bots.groupby(['game_id', 'nickname']).agg(
        avg_points_per_turn=('points', 'mean'),
        avg_extra_points_per_turn=('extra_points', 'mean'),
        avg_word_length=('word_length', 'mean'),
        max_points_in_turn=('points', 'max'),
        bingo_count=('is_bingo', 'sum'),
        hard_letter_plays=('uses_hard_letters', 'sum'),
        pass_count=('is_pass', 'sum'),
        exchange_count=('is_exchange', 'sum'),
        negative_turns_count=('is_negative_turn', 'sum'),
        score=('points', 'sum')
    ).reset_index()

    # Keep only rows in features where 'game_id' exists in train_df_without_bots
    # (i.e., filter features by matching game_ids)
    #features = features[features['game_id'].isin(train_df_without_bots['game_id'])]

    # Filter only bot rows from train_df
    bots_df = train_df[train_df['nickname'].isin(BOTS_NICKNAMES)]

    # Keep only relevant columns and rename 'rating' to 'bot_rating'
    bots_df = bots_df[['game_id', 'rating']].rename(columns={'rating': 'bot_rating'})

    # Merge the bot ratings into features
    features = features.merge(bots_df, on='game_id', how='left')

    # convert 'bot_rating' type from numpy.float64 to numpy.int64
    features['bot_rating'] = features['bot_rating'].fillna(0).astype('int64')


    # Step 4: Add registered user's rating from train_df_without_bots into features
    training_examples_df = pd.merge(features, train_df_without_bots[['game_id', 'nickname', 'rating']],
                                    on=['game_id', 'nickname'])

    # Step 5: Drop 'game_id' & 'nickname' columns
    training_examples_df.drop(columns=['game_id', 'nickname'], inplace=True)

    return training_examples_df
