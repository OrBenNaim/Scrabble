from src.utils import find_best_model, create_dataset

if __name__ == "__main__":
    dataset = create_dataset()

    # Extract rows of training and validation data
    training_examples = dataset[dataset['user_rating'] != 0]

    X_train_val = training_examples.drop(columns=['user_rating'])  # train + validation features df
    y_train_val = training_examples['user_rating']  # Train + validation target vector

    best_model, avg_train_rmse, avg_val_rmse = find_best_model(X_train_val, y_train_val)

    print(best_model, end='\n')
    print(f"RMSE on the training set = {avg_train_rmse:.3f}", end='\n')
    print(f"RMSE on the validation set = {avg_val_rmse:.3f}")

    print(f"Train RMSE: {avg_train_rmse:.3f}")
    print(f"Validation RMSE: {avg_val_rmse:.3f}")

    gap = ((avg_val_rmse / avg_train_rmse) - 1) * 100
    print(f"Generalization Gap: {gap:.3f}%")
