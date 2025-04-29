from src.utils import create_dataset, evaluate_multiple_models, tune_all_models

if __name__ == "__main__":
    dataset = create_dataset()

    # Extract rows of training and validation data
    training_examples = dataset[dataset['user_rating'] != 0]

    X_train_val = training_examples.drop(columns=['user_rating'])  # train + validation features df
    y_train_val = training_examples['user_rating']  # Train + validation target vector

    # Extract rows of testing data
    testing_examples = dataset[dataset['user_rating'] == 0]
    X_test = testing_examples.drop(columns=['user_rating']) # Test features df

    # First evaluate multiple models
    results = evaluate_multiple_models(X_train_val, y_train_val)
    print(results)

    # Tune all models
    cross_val_scores_df, tuned_models = tune_all_models(X_train_val, y_train_val)

    print(tuned_models, end='\n')
    print(cross_val_scores_df, end='\n')
    best_model = tuned_models[cross_val_scores_df.iloc[0]['model']]


    # Make predictions with the tuned model
    #y_pred = tuned_models.predict(X_test)
