import os

# Absolute path to the root of the project (Scrabble/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# DATA folder inside the root
DATA_DIR = os.path.join(PROJECT_ROOT, 'DATA')

# Full paths to your CSV files
GAMES_FILE_PATH = os.path.join(DATA_DIR, 'games.csv')
TURNS_FILE_PATH = os.path.join(DATA_DIR, 'turns.csv')
TRAIN_FILE_PATH = os.path.join(DATA_DIR, 'train.csv')

BOTS_NICKNAMES = ['BetterBot', 'STEEBot', 'HastyBot']
