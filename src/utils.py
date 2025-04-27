import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import (TURNS_FILE_PATH, TRAIN_FILE_PATH, BOTS_NICKNAMES,
                           HARD_LETTERS, SCRABBLE_LETTER_VALUES, BOT_LEVEL_MAPPING, TEST_FILE_PATH,
                           GAMES_FILE_PATH)


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


def create_new_features_from_turns(turns_df_without_bots: pd.DataFrame) -> pd.DataFrame:
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
    turns_df_without_bots : pandas.DataFrame
        Dataframe of Scrabble game turns with at minimum these columns:
        'move', 'points', 'turn_type'

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with added feature columns
    """
    # Precompute all necessary masks and helper columns
    moves = turns_df_without_bots['move'].fillna('')  # Fill NaN with empty string
    turn_types = turns_df_without_bots['turn_type'].str.lower()
    points = turns_df_without_bots['points']

    # Masks and computed columns
    word_lengths = moves.str.len()
    is_bingo = word_lengths >= 7
    has_hard_letter = moves.str.upper().apply(lambda word: any(letter in word for letter in HARD_LETTERS))
    is_pass = turn_types.eq('pass')
    is_exchange = turn_types.eq('exchange')
    is_negative = points < 0

    # Base points calculation
    base_word_points = moves.apply(calculate_base_word_points)
    avg_extra_points_per_game = (points - base_word_points).groupby(turns_df_without_bots['game_id']).mean()

    dataset_df = (
        turns_df_without_bots
        .assign(
            word_length=word_lengths,
            is_bingo=is_bingo,
            has_hard_letter=has_hard_letter,
            is_pass=is_pass,
            is_exchange=is_exchange,
            is_negative=is_negative
        )
        .groupby(['game_id', 'nickname'])
        .agg(
            avg_word_length=('word_length', 'mean'),
            bingo_count=('is_bingo', 'sum'),
            hard_letter_plays=('has_hard_letter', 'sum'),
            negative_turns_count=('is_negative', 'sum'),
            pass_count=('is_pass', 'sum'),
            exchange_count=('is_exchange', 'sum'),
            user_score=('points', 'sum')
        )
        .reset_index()
    )

    # Add avg_extra_points_per_turn
    dataset_df['avg_extra_points_per_turn'] = dataset_df['game_id'].map(avg_extra_points_per_game).fillna(0)

    return dataset_df


def create_new_features_from_train_test(dataset_df: pd.DataFrame, train_df: pd.DataFrame,
                                        test_df: pd.DataFrame):
    # === Create bot_score of all played bots (from train.csv + test.csv) ===

    # Get bots_score_by_game_in_train
    bots_score_by_game_in_train = train_df[train_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['score'].first()

    # Get bots_score_by_game_in_test
    bots_score_by_game_in_test = test_df[test_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['score'].first()

    merged_bots_score = pd.concat([bots_score_by_game_in_train, bots_score_by_game_in_test])

    # Map merged_bots_score to features_df
    dataset_df['bot_score'] = dataset_df['game_id'].map(merged_bots_score).fillna(0).astype('int64')
    #===================================================================================

    # === Create bot_rating of all played bots (from train.csv + test.csv) ===

    # Get bots_rating_by_game_train
    bots_rating_by_game_train = train_df[train_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['rating'].first()  # Take the first bot rating per game (For cases of 2 bots in one game)

    # Get bots_rating_by_game_test
    bots_rating_by_game_test = test_df[test_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['rating'].first()  # Take the first bot rating per game (For cases of 2 bots in one game)

    merged_bots_rating = pd.concat([bots_rating_by_game_train, bots_rating_by_game_test])

    # Map merged_bots_rating to dataset_df
    dataset_df['bot_rating'] = dataset_df['game_id'].map(merged_bots_rating).fillna(0).astype('int64')
    #=========================================================================

    # === Create bot_level of all played bots (from train.csv + test.csv) ===

    # Get the bot_nickname_by_game_train:
    bot_nickname_by_game_train = train_df[train_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['nickname'].first()

    # Get the bot_nickname_by_game_test:
    bot_nickname_by_game_test = test_df[test_df['nickname'].isin(BOTS_NICKNAMES)] \
        .groupby('game_id')['nickname'].first()

    merged_bots_level = pd.concat([bot_nickname_by_game_train, bot_nickname_by_game_test])

    # Map bot_level into dataset_df
    dataset_df['bot_level'] = (dataset_df['game_id'].map(merged_bots_level).map(BOT_LEVEL_MAPPING)
                                .fillna(0).astype('int64'))
    #=========================================================================


def create_dataset():
    """
    Creates a dataset to predict user ratings based on gameplay behavior.

    Steps:
        - Loads game turns data and training data.
        - Excludes bot players from the user data.
        - Filters turns to only include games present in the training set.
        - Creates new gameplay-based features per user and game.
        - Adds:
            * 'bot_rating': the bot's rating in each game.
            * 'bot_level': the bot's level (BetterBot=1, STEEBot=2, HastyBot=3).
            * 'user_rating': the real user's rating in each game.
        - Drops identifiers ('game_id', 'nickname') to keep only features and targets.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a game,
                      with features and the corresponding target rating.
    """

    # === Step 0: Load Data ===
    game_df = pd.read_csv(GAMES_FILE_PATH)
    turns_df = pd.read_csv(TURNS_FILE_PATH)
    train_df = pd.read_csv(TRAIN_FILE_PATH)
    test_df = pd.read_csv(TEST_FILE_PATH)
    #========================================================

    # === Step 1: Set up new dfs from turns_df & train_df ===
    train_df_without_bots = exclude_bots_from_df(train_df)  # Remove bots from train_df

    # Filter out all game_ids that belong to test.csv -> Keep only the rows with game_ids from train_df
    #turns_df = turns_df[turns_df['game_id'].isin(train_df_without_bots['game_id'])]
    turns_df_without_bots = exclude_bots_from_df(turns_df)  # Remove bots from turns_df
    #=====================================================================================================

    # === Step 2: Creates new features for dataset_df ===
    dataset_df = create_new_features_from_turns(turns_df_without_bots=turns_df_without_bots)
    dataset_df['lexicon'] = dataset_df['game_id'].map(game_df.set_index('game_id')['lexicon'])
    create_new_features_from_train_test(dataset_df=dataset_df, train_df=train_df, test_df=test_df)
    #=====================================================================================================

    # === Step 3: Add real user rating per game ===

    # Get the user rating per (game_id, nickname) from train_df_without_bots
    users_rating = train_df_without_bots.set_index(['game_id', 'nickname'])['rating']

    # Map user ratings into features_df
    dataset_df['user_rating'] = (dataset_df.set_index(['game_id', 'nickname']).index.map(users_rating)
                                  .fillna(0).astype('int64'))  # 0 indicates a row that will be predicated later
    #============================================================================================

    # === Step 4: Drop 'game_id' & 'nickname' columns ===
    dataset_df.drop(columns=['game_id', 'nickname'], inplace=True)

    return dataset_df
