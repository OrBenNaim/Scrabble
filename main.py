import pandas as pd

from src.utils import find_best_model, create_dataset
from src.constants import FINAL_PRED_PATH


if __name__ == "__main__":
    dataset = create_dataset()

    # Extract rows of training and validation data
    training_examples = dataset[dataset['user_rating'] != 0]

    X_train_val = training_examples.drop(columns=['user_rating'])  # train + validation features df
    y_train_val = training_examples['user_rating']  # Train + validation target vector

    # Extract rows of testing data
    testing_examples = dataset[dataset['user_rating'] == 0]
    X_test = testing_examples.drop(columns=['user_rating'])  # Test features df

    best_model = find_best_model(X_train_val, y_train_val)

    y_test = best_model.predict(X_test)

    # Convert to pandas DataFrame (allows you to add column names)
    predictions_df = pd.DataFrame()
    predictions_df['game_id'] = X_test.index

    predictions_df['rating'] = y_test
    print(predictions_df)

    # Save to CSV file
    predictions_df.to_csv(FINAL_PRED_PATH, index=False)
