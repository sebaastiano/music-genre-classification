"""Main training script for music genre classification."""
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import (
    load_and_preprocess_data,
    create_sampled_dataset,
    add_synthetic_missing_values,
    create_preprocessing_pipeline
)
from model import (
    create_model_pipeline,
    perform_model_selection,
    fine_tune_best_model
)
from sklearn.metrics import accuracy_score
import numpy as np

warnings.filterwarnings('ignore')


def main():
    """Main training pipeline."""
    print("Loading data...")
    df_original = load_and_preprocess_data('data/dataset.csv')

    print("Creating sampled dataset...")
    df = create_sampled_dataset(
        df_original,
        n_genres=10,
        samples_per_genre=1000,
        random_state=42
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Genres: {df['track_genre'].unique()}")

    print("\nAdding synthetic missing values...")
    df = add_synthetic_missing_values(df)

    print("\nPreparing train/test split...")
    y = df['track_genre']
    X = df.drop(['track_genre'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )

    print("\nCreating preprocessing pipeline...")
    preprocessing_pipeline = create_preprocessing_pipeline()

    print("\nCreating model pipeline...")
    model_pipeline = create_model_pipeline(preprocessing_pipeline)

    print("\nPerforming model selection (this may take a while)...")
    scores = perform_model_selection(model_pipeline, X_train, y_train, cv=5)

    print("\nModel selection results:")
    for idx, score in enumerate(scores['test_score']):
        print(f"Fold {idx+1}: {score:.4f}")

    best_estimator_index = np.argmax(scores['test_score'])
    best_estimator = scores['estimator'][best_estimator_index].best_estimator_

    print(f"\nBest model accuracy: {scores['test_score'][best_estimator_index]:.4f}")
    print(f"Best estimator configuration:")
    print(f"  Sampler: {best_estimator.get_params()['sampler']}")
    print(f"  Dim reduction: {best_estimator.get_params()['dim_reduction']}")
    print(f"  Classifier: {best_estimator.get_params()['classifier']}")

    print("\nFine-tuning best model...")
    best_pipeline = create_model_pipeline(preprocessing_pipeline)
    rs_best = fine_tune_best_model(best_pipeline, X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = rs_best.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    # Calculate training accuracy
    y_train_pred = rs_best.best_estimator_.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Final Training Accuracy: {train_accuracy:.4f}")

    print("\nBest hyperparameters:")
    print(rs_best.best_params_)

    return rs_best, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = main()