# This model uses the RandomForestRegressor
import argparse
import sys
from os import path

import numpy
import pandas
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATASET_FILE_PATH = path.join(path.dirname(path.dirname(__file__)), 'datasets/winequality-white.csv')
MODELS_DIR_PATH = path.join(path.dirname(path.dirname(__file__)), 'models')
CORRELATION_THRESHOLD = .05
MODELS = ['rfc', 'sgdc', 'svc', 'lr', 'rfr', 'abc', 'dtc']
CLASSIFICATION_MODELS = ['rfc', 'sgdc', 'svc', 'abc', 'dtc']
REGRESSION_MODELS = ['lr', 'rfr']
WINE_QUALITY_LABELS = ['bad', 'good']

MODEL_ARG_HELP_TEXT = '''
Specifies the model to be generated
    OPTIONS
    - rfc - Random Forest Classifier
    - sgdc - Stochastic Gradient Descent Classifier
    - svc - Support Vector Classifier
    - ada - Ada Booster Classifier
    - lr - Linear Regressor
    - rfr - Random Forest Regressor
'''


def get_training_correlations(wine_quality_data=None):
    if wine_quality_data is None:
        wine_quality_data = pandas.read_csv(DATASET_FILE_PATH, sep=';')
    return wine_quality_data.corr()['quality'].drop('quality')


def record_accuracy(model, accuracy):
    ACCURACY_RECORDS_FILE_PATH = path.join(MODELS_DIR_PATH, 'accuracy_records.jbl')
    accuracy_records = None
    try:
        accuracy_records = load(ACCURACY_RECORDS_FILE_PATH)
    except FileNotFoundError:
        accuracy_records = {}

    accuracy_records[model] = accuracy
    dump(accuracy_records, ACCURACY_RECORDS_FILE_PATH)


def get_relevant_features(abs_correlations):
    return abs_correlations[abs_correlations > CORRELATION_THRESHOLD].index.values.tolist()


def preprocess(problem_type='classification'):
    # Load data
    wine_quality_data = pandas.read_csv(DATASET_FILE_PATH, sep=';')
    relevant_features = get_relevant_features(get_training_correlations(wine_quality_data).abs())
    x = wine_quality_data[relevant_features]
    bins = [2, 6.5, 8]
    y = None

    if problem_type == 'classification':
        quality_group_labels = ['bad', 'good']
        wine_quality_data['quality_group'] = pandas.cut(wine_quality_data['quality'], bins=bins,
                                                        labels=quality_group_labels)
        label_encoder = LabelEncoder()
        wine_quality_data['quality_group'] = label_encoder.fit_transform(wine_quality_data['quality_group'].astype(str))
        y = wine_quality_data['quality_group']
    elif problem_type == 'regression':
        y = wine_quality_data['quality']
    else:
        raise Exception('Invalid Problem Type')

    return train_test_split(x, y, test_size=0.2, random_state=42)


def generate_rfc_model(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model = RandomForestClassifier(n_estimators=200)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    dump(model, path.join(MODELS_DIR_PATH, 'rfc_model.jbl'))

    # record_accuracy()


def generate_sgdc_model(x_train, x_test, y_train, y_test):
    model = SGDClassifier(penalty=None)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    dump(model, path.join(MODELS_DIR_PATH, 'sgdc_model.jbl'))


def generate_svc_model(x_train, x_test, y_train, y_test):
    model = SVC(C=1.2, gamma=3.5, kernel='rbf')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    dump(model, path.join(MODELS_DIR_PATH, 'svc_model.jbl'))


def generate_lr_model(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print('Accuracy ==> ', 100 - round(r2_score(y_test, numpy.round(y_pred)) * 100, 2))

    dump(model, path.join(MODELS_DIR_PATH, 'lr_model.jbl'))


def generate_rfr_model(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print('Accuracy ==> ', 100 - round(r2_score(y_test, numpy.round(y_pred)) * 100, 2))

    dump(model, path.join(MODELS_DIR_PATH, 'rfr_model.jbl'))


def generate_dtc_model(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    dump(model, path.join(MODELS_DIR_PATH, 'dtc_model.jbl'))


def generate_abc_model(x_train, x_test, y_train, y_test):
    model = AdaBoostClassifier(random_state=30)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    dump(model, path.join(MODELS_DIR_PATH, 'rfr_model.jbl'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Wine Quality Prediction Model Trainer')
    arg_parser.add_argument('--model', type=str, help=MODEL_ARG_HELP_TEXT, choices=MODELS)
    args = arg_parser.parse_args()

    x_train, x_test, y_train, y_test = preprocess(
        problem_type='regression' if args.model in REGRESSION_MODELS else 'classification')

    getattr(sys.modules[__name__], f'generate_{args.model}_model')(x_train, x_test, y_train, y_test)
