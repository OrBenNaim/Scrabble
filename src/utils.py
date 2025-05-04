import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_validate

from src.constants import (GAMES_FILE_PATH, TURNS_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, BOTS_NICKNAMES,
                           HARD_LETTERS, SCRABBLE_LETTER_VALUES, BOT_LEVEL_MAPPING, CV_N_SPLITS, N_TRIALS,
                           MODEL_CONFIGS)


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
    """
    Enhances the provided dataset with bot-related features derived from training and test datasets.

    This function adds three bot-related columns to the dataset_df:
        - bot_score: The score achieved by bots in each game
        - bot_rating: The rating of bots in each game
        - bot_level: The level of bots (1, 2, or 3) based on their type (mapped via BOT_LEVEL_MAPPING)

    For games with multiple bots, the first bot's information is used. Missing values are filled with 0.

    Parameters:
    -----------
        dataset_df : pandas.DataFrame
            The dataset to enhance with bot features. Must contain a 'game_id' column.
        train_df : pandas.DataFrame
            Training dataset containing game information. Must have 'game_id', 'nickname',
            'score', and 'rating' columns.
        test_df : pandas.DataFrame
            Test dataset containing game information. Must have the same structure as train_df.

    Returns:
    --------
        pandas.DataFrame
            The original dataset_df with added bot feature columns.

    Notes:
    ------
        - BOTS_NICKNAMES is expected to be a list of bot nicknames used to identify bots in the data
        - BOT_LEVEL_MAPPING is expected to be a dictionary mapping bot types to their level (1, 2, or 3)
    """

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
    Creates a dataset with all features and 'y' target (dataset contains also testing data)¶

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
    turns_df_without_bots = exclude_bots_from_df(turns_df)  # Remove bots from turns_df
    #=====================================================================================================

    # === Step 2: Creates new features for dataset_df ===
    dataset_df = create_new_features_from_turns(turns_df_without_bots=turns_df_without_bots)
    dataset_df['lexicon'] = dataset_df['game_id'].map(game_df.set_index('game_id')['lexicon'])
    create_new_features_from_train_test(dataset_df=dataset_df, train_df=train_df, test_df=test_df)

    # Adding new features in False Analysis phase

    # Measures how much the user outperformed (or underperformed) the bot — often directly correlated with target rating
    dataset_df['score_diff'] = dataset_df['user_score'] - dataset_df['bot_score']

    # Represents high-risk, high-reward play styles — long words with extra points
    dataset_df['aggression_score'] = dataset_df['avg_extra_points_per_turn'] * dataset_df['avg_word_length']

    # Normalizes bingo's by overall score — helpful for measuring how dependent a player is on bingo's
    dataset_df['bingo_density'] = dataset_df['bingo_count'] / (dataset_df['user_score'] + 1e-5)

    # Normalizes score by word complexity; useful to reward concise but effective play
    dataset_df['efficiency_score'] = dataset_df['user_score'] / (dataset_df['avg_word_length'] + 1e-5)

    # Detects when the bot played unusually well or poorly compared to its expected rating
    dataset_df['bot_rating_diff'] = dataset_df['bot_rating'] - dataset_df['bot_score']

    # How effective each bingo was — high values mean other plays contributed more than bingo's
    dataset_df['bingo_efficiency'] = dataset_df['user_score'] / (dataset_df['bingo_count'] + 1)

    # Measures how often the player used challenging letters (Q, Z, X, etc.) — may correlate with skill or desperation
    dataset_df['hard_letter_rate'] = dataset_df['hard_letter_plays'] / (dataset_df['user_score'] + 1e-5)

    # Measures whether the bot played above or below its expected level
    dataset_df['bot_performance_ratio'] = dataset_df['bot_score'] / (dataset_df['bot_rating'] + 1e-5)
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


def tune_all_models(X_train_val: pd.DataFrame, y_train_val: pd.Series):
    """
    Tune hyperparameters for Random Forest, XGBoost, and LightGBM using Bayesian Optimization.

    Parameters:
    -----------
    X_train_val : pandas.DataFrame
        Training features.
    y_train_val : pandas.Series
        Training target values.

    Returns:
    --------
    scores_df : pandas.DataFrame
        DataFrame with columns: ['model', 'neg_rmse'] for each model.
    best_models : dict
        Dictionary of model names mapped to their optimized pipelines.
    """

    # Extract feature types
    # Identify feature types
    numerical_features = X_train_val.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train_val.select_dtypes(exclude=['number']).columns.tolist()

    # Create shared preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

    best_models = {}
    scores = []

    for model_name, model_builder in MODEL_CONFIGS.items():
        print(f"\nOptimizing: {model_name}")

        def objective(trial):
            current_model = model_builder(trial)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', current_model)
            ])
            score = cross_val_score(estimator=pipeline, X=X_train_val, y=y_train_val, cv=CV_N_SPLITS,
                                    scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
            return -score  # negative RMSE

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS)

        best_score = study.best_value
        best_params = study.best_trial.params

        print(f"Best score for {model_name}: {best_score:.4f}")
        print(f"\nBest params: {best_params}")

        # Build final model with best params
        model = model_builder(study.best_trial)
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        best_models[model_name] = final_pipeline
        scores.append({'model': model_name, 'Mean_CV_RMSE': best_score})

    scores_df = pd.DataFrame(scores).sort_values(by='Mean_CV_RMSE', ascending=True).reset_index(drop=True)
    return scores_df, best_models


def find_best_model(X_train_val: pd.DataFrame, y_train_val: pd.Series) -> (Pipeline, float, float):
    """
    Tune multiple models, select the one with the best cross-validated performance, and compute train/validation RMSE.

    Parameters:
    -----------
    X_train_val : pd.DataFrame
        Combined training and validation features.
    y_train_val : pd.Series
        Combined training and validation target values.

    Returns:
    --------
    best_model : sklearn.pipeline.Pipeline
        The tuned model pipeline with the lowest validation RMSE.
    avg_train_rmse : float
        Average RMSE on the training folds during cross-validation.
    avg_val_rmse : float
        Average RMSE on the validation folds during cross-validation.
    """
    # Tune all models
    cross_val_scores_df, tuned_models = tune_all_models(X_train_val, y_train_val)

    print(cross_val_scores_df, end='\n')
    print(tuned_models, end='\n')

    best_model = tuned_models[cross_val_scores_df.iloc[0]['model']]

    avg_train_rmse, avg_val_rmse = calc_train_val_rmse_cv(best_model, X_train_val, y_train_val)

    return best_model, avg_train_rmse, avg_val_rmse


def calc_train_val_rmse_cv(pipeline: Pipeline, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> (float, float):
    """
    Evaluate a model pipeline using cross-validation and return RMSE for training and validation sets.

    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        The model pipeline to evaluate.
    X_train_val : pd.DataFrame
        Combined training and validation features.
    y_train_val : pd.Series
        Combined training and validation target values.

    Returns:
    --------
    avg_train_rmse : float
        Average RMSE on the training folds.
    avg_val_rmse : float
        Average RMSE on the validation folds.
    """

    results = cross_validate(
        estimator=pipeline,
        X=X_train_val,
        y=y_train_val,
        cv=CV_N_SPLITS,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        n_jobs=-1
    )

    train_scores = results['train_score']  # negative RMSE on training folds
    val_scores = results['test_score']  # negative RMSE on validation folds

    avg_train_rmse = -train_scores.mean()  # Convert to RMSE (positive values)
    avg_val_rmse = -val_scores.mean()

    return avg_train_rmse, avg_val_rmse
