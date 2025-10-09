"""Model training and evaluation utilities."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, learning_curve,
    validation_curve, cross_validate
)
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import itertools


def create_model_pipeline(preprocessing_pipeline):
    """
    Create the model pipeline with preprocessing and classifier.

    Parameters:
    -----------
    preprocessing_pipeline : ColumnTransformer
        Preprocessing pipeline

    Returns:
    --------
    Pipeline
        Complete model pipeline
    """
    return Pipeline([
        ('trans', preprocessing_pipeline),
        ('sampler', None),
        ('dim_reduction', None),
        ('classifier', OneVsRestClassifier(LinearSVC()))
    ])


def get_search_configurations():
    """
    Get hyperparameter search configurations.

    Returns:
    --------
    list
        List of all possible configurations
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.random_projection import SparseRandomProjection
    from sklearn.multiclass import OneVsOneClassifier

    sampler_configs = [{'sampler': [None]}]

    dim_reduction_configs = [
        {'dim_reduction': [None]},
        {
            'dim_reduction': [TruncatedSVD()],
            'dim_reduction__n_components': [10, 16, 100],
            'dim_reduction__algorithm': ['arpack', 'randomized']
        },
        {
            'dim_reduction': [NMF()],
            'dim_reduction__n_components': [10, 50, 100],
            'dim_reduction__init': ['random', 'nndsvd', 'nndsvda']
        },
        {
            'dim_reduction': [SparseRandomProjection()],
            'dim_reduction__n_components': [10, 50, 100, 'auto'],
            'dim_reduction__dense_output': [True, False]
        }
    ]

    estimator_configs = [
        {
            'classifier': [KNeighborsClassifier()],
            'classifier__n_neighbors': [87, 89, 91, 93],
            'classifier__weights': ['uniform', 'distance']
        },
        {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [10, 50, 100, 500],
            'classifier__bootstrap': [True, False]
        },
        {
            'classifier': [OneVsOneClassifier(LinearSVC())],
            'classifier__estimator__C': [0.1, 1.0, 10.0],
            'classifier__estimator__penalty': ['l1', 'l2'],
            'classifier__estimator__dual': ['auto'],
            'classifier__estimator__max_iter': [100000],
            'classifier__n_jobs': [-1]
        },
        {
            'classifier': [OneVsRestClassifier(LinearSVC())],
            'classifier__estimator__C': [0.1, 1.0, 10.0],
            'classifier__estimator__penalty': ['l1', 'l2'],
            'classifier__estimator__dual': ['auto'],
            'classifier__estimator__max_iter': [100000],
            'classifier__n_jobs': [-1]
        }
    ]

    all_configs = []
    for config in itertools.product(sampler_configs, dim_reduction_configs, estimator_configs):
        all_parameters = []
        for element in config:
            for item in element.items():
                all_parameters.append(item)
        all_configs.append(dict(all_parameters))

    return all_configs


def perform_model_selection(pipeline, X_train, y_train, cv=5):
    """
    Perform model selection using RandomizedSearchCV.

    Parameters:
    -----------
    pipeline : Pipeline
        Model pipeline
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    dict
        Cross-validation scores and estimators
    """
    all_configs = get_search_configurations()

    rs = RandomizedSearchCV(
        pipeline,
        param_distributions=all_configs,
        n_iter=len(all_configs) * 5,
        n_jobs=-1,
        cv=2,
        scoring='accuracy'
    )

    scores = cross_validate(
        rs, X_train, y_train,
        scoring='accuracy',
        cv=cv,
        return_estimator=True,
        verbose=3
    )

    return scores


def fine_tune_best_model(best_pipeline, X_train, y_train):
    """
    Fine-tune the best model with hyperparameter optimization.

    Parameters:
    -----------
    best_pipeline : Pipeline
        Best model pipeline from model selection
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels

    Returns:
    --------
    RandomizedSearchCV
        Fitted RandomizedSearchCV object with best estimator
    """
    params = {
        'classifier__estimator__C': [1.0, 10.0],
        'classifier__estimator__penalty': ['l1', 'l2'],
        'classifier__estimator__dual': ['auto'],
        'classifier__estimator__max_iter': [100000],
        'classifier__n_jobs': [-1]
    }

    rs_best = RandomizedSearchCV(
        estimator=best_pipeline,
        param_distributions=params,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
        n_iter=20,
        scoring='accuracy'
    )

    rs_best.fit(X_train, y_train)
    return rs_best


def get_learning_curve_data(model, X_train, y_train):
    """
    Generate learning curve data.

    Parameters:
    -----------
    model : estimator
        Trained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels

    Returns:
    --------
    tuple
        train_sizes, train_scores, test_scores
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X=X_train,
        y=y_train,
        train_sizes=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        shuffle=False
    )

    return train_sizes, train_scores, test_scores


def get_validation_curve_data(model, X_train, y_train, param_range):
    """
    Generate validation curve data.

    Parameters:
    -----------
    model : estimator
        Trained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    param_range : list
        Range of parameter values to test

    Returns:
    --------
    tuple
        train_scores, test_scores
    """
    train_scores, test_scores = validation_curve(
        model,
        X=X_train,
        y=y_train,
        param_range=param_range,
        param_name='classifier__estimator__C',
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    return train_scores, test_scores