import os

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