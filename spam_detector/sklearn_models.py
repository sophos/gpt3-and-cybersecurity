import pandas as pd
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(
    df,
    column_text="text",
    column_label="label",
    positive_label="Spam",
    max_features=1000,
    vectorizer=None,
):
    """
    extract ML features.
    :param df: the data frame for input data
    :param column_text: the column for text
    :param column_label: the column for label
    :param positive_label: the value for positive label
    :param max_features: the max number of features
    :param vectorizer: vectorizer for test data
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df[column_text])
    else:
        X = vectorizer.transform(df[column_text])
    y = df[column_label] == positive_label
    return X, y, vectorizer


def train_sk_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name="RandomForest"
):
    if model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True)


def evaluate_sklearn_model(
    path_train_data,
    path_test_data,
    model_name="RandomForest",
    max_features=1000,
    column_text="text",
    column_label="label",
    positive_label="Spam"
):
    """
    evaluate sklearn models with training and test datasets.
    :param path_train_data: file path for training dataset
    :param path_test_data: file path for test dataset
    :param model_name: model name
    :param max_features: max number of ML features
    :param column_text: the column for text
    :param column_label: the column for label
    :param positive_label: the value for positive label
    """
    df_train = pd.read_csv(path_train_data, sep="\t")
    logger.info("path_train_data:{}, df_train.shape:{}".format(
        path_train_data, df_train.shape))

    X_train, y_train, vectorizer = extract_features(
        df_train, max_features=max_features,
        column_text=column_text, column_label=column_label,
        positive_label=positive_label)
    logger.info("X_train.shape:{}, y_train.shape:{}".format(
        X_train.shape, y_train.shape))
    logger.info("y_train.label.count:{}".format(Counter(y_train)))

    df_test = pd.read_csv(path_test_data, sep="\t")
    logger.info("path_test_data:{}, df_test.shape:{}".format(
        path_test_data, df_test.shape))

    X_test, y_test, _vectorizer = extract_features(
        df_test, max_features=max_features, vectorizer=vectorizer,
        column_text=column_text, column_label=column_label,
        positive_label=positive_label)
    logger.info("X_test.shape:{}, y_test.shape:{}".format(
        X_test.shape, y_test.shape))
    logger.info("y_test.label.count:{}".format(Counter(y_test)))

    return train_sk_model(X_train, X_test, y_train, y_test, model_name=model_name)
