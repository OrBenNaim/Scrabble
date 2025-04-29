from sklearn.model_selection import train_test_split

from src.utils import create_dataset, evaluate_multiple_models, tune_best_model

if __name__ == "__main__":
    dataset = create_dataset()

    # Extract rows of training and validation data
    training_examples = dataset[dataset['user_rating'] != 0]

    X_train_val = training_examples.drop(columns=['user_rating'])  # train + validation fatures df
    y_train_val = training_examples['user_rating']  # Train + validation target vector

    # Extract rows of testing data
    testing_examples = dataset[dataset['user_rating'] == 0]
    X_test = testing_examples.drop(columns=['user_rating']) # Test features df

    # First evaluate multiple models
    results = evaluate_multiple_models(X_train_val, y_train_val)
    print(results)

    # Choose the best model based on validation score
    best_model_name = results.iloc[0]['Model']
    print(f"Best model: {best_model_name}")

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Tune the best model
    tuned_model, best_params = tune_best_model(X_train, y_train, best_model_name)

    # Make predictions with the tuned model
    y_pred = tuned_model.predict(X_test)
