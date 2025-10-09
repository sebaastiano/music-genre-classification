"""Data preprocessing utilities for music genre classification."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


def add_missing(col, amount):
    """
    Add missing values to a column for demonstration purposes.

    Parameters:
    -----------
    col : pd.Series
        The column to add missing values to
    amount : float or int
        If >= 1, number of values to set as NaN
        If < 1, proportion of values to set as NaN

    Returns:
    --------
    pd.Series
        Column with added missing values
    """
    X = col.copy()
    size = amount if amount >= 1 else int(len(X) * amount)
    indexes = np.random.choice(len(X), size, replace=False)
    X[indexes] = np.nan
    return X


def load_and_preprocess_data(file_path, random_state=42):
    """
    Load and preprocess the Spotify dataset.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(['Unnamed: 0', 'track_id', 'album_name', 'track_name'],
            axis=1, inplace=True)

    # Convert boolean to int
    df['explicit'] = df['explicit'].map({False: 0, True: 1})

    return df


def create_sampled_dataset(df, n_genres=10, samples_per_genre=1000, random_state=42):
    """
    Create a sampled dataset with specified number of genres and samples.

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    n_genres : int
        Number of genres to sample
    samples_per_genre : int
        Number of samples per genre
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    pd.DataFrame
        Sampled and shuffled dataframe
    """
    np.random.seed(random_state)
    selected_genres = np.random.choice(df['track_genre'].unique(),
                                      size=n_genres, replace=False)

    sampled_instances = [df[df['track_genre'] == genre].copy()
                        for genre in selected_genres]
    df_sampled = pd.concat(sampled_instances)
    df_sampled = df_sampled.sample(frac=1, random_state=random_state)

    df_final = pd.concat([group.sample(n=samples_per_genre, random_state=random_state)
                         for _, group in df_sampled.groupby('track_genre')])
    df_final = df_final.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_final


def add_synthetic_missing_values(df):
    """
    Add synthetic missing values to demonstrate imputation.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to add missing values to

    Returns:
    --------
    pd.DataFrame
        Dataframe with added missing values
    """
    df['danceability'] = add_missing(df['danceability'], 0.02)
    df['energy'] = add_missing(df['energy'], 0.05)
    df['liveness'] = add_missing(df['liveness'], 0.05)
    df['valence'] = add_missing(df['valence'], 0.08)
    df['time_signature'] = add_missing(df['time_signature'], 0.08)

    # Drop rows with missing artists or too many missing values
    df.dropna(subset=['artists'], inplace=True)
    df.dropna(thresh=3, inplace=True)

    return df


def create_preprocessing_pipeline():
    """
    Create the preprocessing pipeline for the dataset.

    Returns:
    --------
    ColumnTransformer
        Preprocessing pipeline
    """
    pipe_mean_std = Pipeline([
        ('Mean', SimpleImputer(strategy='mean')),
        ('Scaler', StandardScaler())
    ])

    pipe_median_std = Pipeline([
        ('Median', SimpleImputer(strategy='median')),
        ('Scaler', StandardScaler())
    ])

    pipe_mode = Pipeline([
        ('Mode', SimpleImputer(strategy='most_frequent'))
    ])

    song_transformer = ColumnTransformer(
        transformers=[
            ('artists', OneHotEncoder(handle_unknown='ignore'), ['artists']),
            ('danceability', pipe_mean_std, ['danceability', 'energy', 'liveness']),
            ('valence', pipe_median_std, ['valence']),
            ('time_signature', pipe_mode, ['time_signature']),
            ('minmax_scaling', MinMaxScaler(),
             ['popularity', 'loudness', 'duration_ms', 'tempo'])
        ]
    )

    return song_transformer