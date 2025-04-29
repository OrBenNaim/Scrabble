import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Absolute path to the root of the project (Scrabble/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# DATA folder inside the root
DATA_DIR = os.path.join(PROJECT_ROOT, 'DATA')

# Full paths to your CSV files
GAMES_FILE_PATH = os.path.join(DATA_DIR, 'games.csv')
TURNS_FILE_PATH = os.path.join(DATA_DIR, 'turns.csv')
TRAIN_FILE_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(DATA_DIR, 'test.csv')

BOTS_NICKNAMES = ['BetterBot', 'STEEBot', 'HastyBot']
BOT_LEVEL_MAPPING = { 'BetterBot': 1, 'STEEBot': 2, 'HastyBot': 3 }

HARD_LETTERS = ['J', 'K', 'Q', 'X', 'Z']    # Define hard Scrabble letters (high-point tiles)

SCRABBLE_LETTER_VALUES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2,
    'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1,
    'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10,'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10
}


# === Hyperparameter Search Space Constants ===
RANGE_N_ESTIMATORS = (100, 500)
RANGE_MAX_DEPTH_RF = (5, 50)
RANGE_MAX_DEPTH_GBM = (3, 12)
RANGE_MIN_SAMPLES_SPLIT = (2, 20)
RANGE_LEARNING_RATE = (0.01, 0.3)
RANGE_SUBSAMPLE = (0.5, 1.0)
RANGE_NUM_LEAVES = (20, 150)
CV_N_SPLITS = 5     # This parameter in KFold cross-validation defines how many parts the dataset will be split into
N_TRIALS = 10       # This parameter in study.optimize() defines  number of optimization
                    # iterations (trials) the tuner will run.

MODEL_CONFIGS = {
        'Random Forest': lambda trial: RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', *RANGE_N_ESTIMATORS),
            max_depth=trial.suggest_int('max_depth', *RANGE_MAX_DEPTH_RF),
            min_samples_split=trial.suggest_int('min_samples_split', *RANGE_MIN_SAMPLES_SPLIT),
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': lambda trial: XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', *RANGE_N_ESTIMATORS),
            max_depth=trial.suggest_int('max_depth', *RANGE_MAX_DEPTH_GBM),
            learning_rate=trial.suggest_float('learning_rate', *RANGE_LEARNING_RATE, log=True),
            subsample=trial.suggest_float('subsample', *RANGE_SUBSAMPLE),
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lambda trial: LGBMRegressor(
            n_estimators=trial.suggest_int('n_estimators', *RANGE_N_ESTIMATORS),
            max_depth=trial.suggest_int('max_depth', *RANGE_MAX_DEPTH_GBM),
            learning_rate=trial.suggest_float('learning_rate', *RANGE_LEARNING_RATE, log=True),
            num_leaves=trial.suggest_int('num_leaves', *RANGE_NUM_LEAVES),
            random_state=42,
            n_jobs=-1
        )
    }
