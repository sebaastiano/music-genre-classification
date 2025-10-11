# Music Genre Classification

A machine learning project that predicts music genres from Spotify data using various audio features and metadata.

## Overview

This project was developed for the final exam of the Machine Learning course in the Bachelor in Artificial Intelligence. The model is capable of predicting the genre of a song based on features from the music domain (tempo, key, time signature), lyrics domain (speechiness, explicitness), and public perception metrics (popularity, danceability). All data is extracted from Spotify's database.

## Features

The model uses the following features for prediction:

- **artists**: Artist names who performed the track
- **popularity**: Track popularity (0-100)
- **explicit**: Whether the track has explicit lyrics
- **danceability**: Suitability for dancing based on musical elements
- **energy**: Perceptual measure of intensity and activity
- **key**: Musical key of the track
- **loudness**: Overall loudness in decibels (dB)
- **mode**: Modality (major or minor)
- **speechiness**: Presence of spoken words
- **acousticness**: Confidence measure of acoustic content
- **instrumentalness**: Prediction of vocal absence
- **liveness**: Presence of audience in recording
- **valence**: Musical positiveness
- **tempo**: Estimated tempo in BPM
- **time_signature**: Estimated time signature (3/4 to 7/4)

## Installation

Clone the repository:
```bash
git clone https://github.com/sebaleye/music-genre-classification.git
cd music-genre-classification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Download the Spotify Tracks Dataset from Kaggle:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place the `dataset.csv` file in the `data/` directory.

## Usage

### Training the Model

Run the main training script:

```bash
cd src
python train.py
```

This will:
- load and preprocess the data
- build a balanced dataset with 10 random genres (1000 samples each)
- add synthetic missing values to demonstrate imputation
- perform model selection using cross-validation
- fine-tune the best model
- evaluate on the test set

### Using the Jupyter Notebook

For interactive exploration and visualization:

```bash
jupyter notebook notebooks/genre_ml.ipynb
```

### Custom Training

You can modify the training parameters in `train.py`:

```python
df = create_sampled_dataset(
    df_original,
    n_genres=10,              # number of genres to include
    samples_per_genre=1000,   # samples per genre
    random_state=42
)
```

## Model Performance

The best performing model achieved:
- **Test Accuracy**: ~92.65%
- **Training Accuracy**: ~99.95%

The model uses:
- **Classifier**: One-vs-Rest LinearSVC
- **Preprocessing**: StandardScaler, MinMaxScaler, One-Hot Encoding
- **No dimensionality reduction** (data naturally low-dimensional)
- **No sampling** (dataset is balanced by construction)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- scipy
- mlxtend
- imbalanced-learn
- missingno
- jupyter

See `requirements.txt` for specific versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya

## Contact

For questions or feedback, please open an issue on GitHub.
