"""Visualization utilities for data analysis and results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import missingno as msno


def plot_genre_distribution(df):
    """
    Plot histogram of genre distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with track_genre column
    """
    histogram = sns.catplot(data=df, x="track_genre", kind="count", aspect=1.5)
    histogram.set_xticklabels([])
    histogram.ax.yaxis.grid(True)
    histogram.ax.set_axisbelow(True)
    plt.title('Genre Distribution')
    plt.show()


def plot_missing_values_matrix(df):
    """
    Plot missing values matrix.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    """
    msno.matrix(df)
    plt.show()


def plot_feature_by_genre(df):
    """
    Plot features by genre in subplots.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with genres and features
    """
    df_noartists = df.drop(['artists'], axis=1)
    df_genres = df_noartists.groupby(['track_genre'], axis=0, as_index=False).mean()

    fig, axes = plt.subplots(4, 4, figsize=(25, 25))
    axes = axes.flatten()

    for i, column in enumerate(df_genres.columns[1:]):
        data = pd.DataFrame(data={
            'genre': df_genres['track_genre'],
            'value': df_genres[column]
        })

        if column == 'loudness':
            data['value'] = data['value'].abs()
            data = data.sort_values('value')
        else:
            data = data.sort_values('value', ascending=False)

        ax = sns.barplot(x="genre", y='value', data=data, ci=None, ax=axes[i])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=12)
        ax.set_title(column.capitalize(), fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y')
        ax.set_axisbelow(True)

    for i in range(len(df_genres.columns[1:]), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot correlation matrix including encoded genre columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features and genres
    """
    df_noartists = df.drop(['artists'], axis=1)
    df_genres = df_noartists.groupby(['track_genre'], axis=0, as_index=False).mean()

    numerical_columns = [
        'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]

    df_encoded = pd.get_dummies(df_genres, columns=['track_genre'])
    encoded_columns = df_encoded.columns[df_encoded.columns.str.startswith('track_genre_')]

    df_concatenated = pd.concat([df_genres[numerical_columns], df_encoded[encoded_columns]], axis=1)
    correlation_matrix = df_concatenated.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix (Including Genre Encoded Columns)")
    plt.show()


def plot_scoring_comparison(scores_dict):
    """
    Plot comparison of different scoring methods.

    Parameters:
    -----------
    scores_dict : dict
        Dictionary with scoring method names as keys and scores as values
    """
    mean_scores = {method: np.mean(score) for method, score in scores_dict.items()}

    plt.figure(figsize=(10, 6))
    bars = plt.bar(mean_scores.keys(), mean_scores.values(),
                   color=['#FF6F61', '#85C1E9', '#3498DB', '#21618C'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3),
                ha='center', va='bottom', color='black')

    plt.title('Comparison of Scoring Methods', fontsize=16)
    plt.xlabel('Scoring Method', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_learning_curve(train_sizes, train_scores, test_scores):
    """
    Plot learning curve.

    Parameters:
    -----------
    train_sizes : array
        Training set sizes
    train_scores : array
        Training scores
    test_scores : array
        Validation scores
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()

    ax.plot(train_sizes, train_mean, color='blue', marker='+',
            markersize=5, label='Training accuracy')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                    alpha=0.15, color='blue')

    ax.plot(train_sizes, test_mean, color='green', linestyle='--',
            marker='d', markersize=5, label='Validation accuracy')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                    alpha=0.15, color='green')

    ax.grid()
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Learning Curve')
    ax.legend(loc='lower right')
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix.

    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cm_display = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, normalize='true', ax=ax, cmap='Blues'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_validation_curve(param_range, train_scores, test_scores):
    """
    Plot validation curve.

    Parameters:
    -----------
    param_range : array
        Parameter values
    train_scores : array
        Training scores
    test_scores : array
        Validation scores
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()

    ax.plot(param_range, train_mean, color='blue', marker='o',
            markersize=5, label='Training accuracy')
    ax.fill_between(param_range, train_mean + train_std, train_mean - train_std,
                    alpha=0.15, color='blue')

    ax.plot(param_range, test_mean, color='green', linestyle='--',
            marker='s', markersize=5, label='Validation accuracy')
    ax.fill_between(param_range, test_mean + test_std, test_mean - test_std,
                    alpha=0.15, color='green')

    ax.grid()
    ax.set_xlabel('Parameter C')
    ax.set_ylabel('Score')
    ax.set_title('Validation Curve')
    ax.legend(loc='lower right')
    ax.set_xscale('log')
    plt.show()