import argparse
from os import path

import pandas
from joblib import load

from src.train import get_relevant_features, get_training_correlations, MODELS, MODELS_DIR_PATH, \
    CLASSIFICATION_MODELS, WINE_QUALITY_LABELS, MODEL_ARG_HELP_TEXT


def collect_and_shape_data():
    relevant_features = get_relevant_features(get_training_correlations().abs())

    data = {}
    # for feature in relevant_features:
    #     data[feature] = (float(input(f'{feature} = ')))

    data = {'fixed acidity': 7.1, 'volatile acidity': 0.875, 'residual sugar': 5.7, 'chlorides': 0.082,
            'total sulfur dioxide': 14, 'density': 0.99808, 'pH': 3.4, 'sulphates': 0.52, 'alcohol': 10.2}
    return pandas.DataFrame(data, relevant_features)


if __name__ == '__main__':
    print(MODELS_DIR_PATH)
    arg_parser = argparse.ArgumentParser(description='Wine Quality Predictor')
    arg_parser.add_argument('--model', type=str, help=MODEL_ARG_HELP_TEXT, choices=MODELS)
    args = arg_parser.parse_args()

    data = collect_and_shape_data()

    model = None

    try:
        model = load(path.join(MODELS_DIR_PATH, f'{args.model}_model.jbl'))
    except FileNotFoundError:
        raise SystemExit(f'Run `python train.py --model {args.model} to generate the train the model!')

    y_pred = model.predict(data)

    if args.model in CLASSIFICATION_MODELS:
        print('Predicted Quality ==>', WINE_QUALITY_LABELS[y_pred[0]])
    else:
        print('Predicted Quality on scale [0 - 10] ==>', round(y_pred[0]))
