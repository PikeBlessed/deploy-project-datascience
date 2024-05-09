"""
train.py: Script for training a machine learning model.

This script loads data, loads a pre-trained model, performs cross-validation,
calculates training and test scores, and logs the results.

Usage:
    python train.py

Author:
    Octavio

"""

import os
from sklearn.model_selection import cross_validate, train_test_split
from model_utils import update_model, save_simple_metrics_report, get_model_performance
import logging
import sys
import pandas as pd

from joblib import load

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate the training process.
    """
    try:
        data_path = input("Enter the path to the CSV file containing the data: ")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"File '{data_path}' not found.")
        
        model_path = input("Enter the path to the pre-trained model file: ")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File '{model_path}' not found.")
         
        logger.info('Loading data...')
        df = pd.read_csv(data_path)

        logger.info('Loading model...')
        model = load(model_path)

        logger.info('Separating dataset into train and test')
        X = df.drop(['reach', 'date', 'engagement'], axis=1)
        y = df['reach']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=42)
        model.fit(X_train, y_train)

        logger.info('Cross validating with best model...')
        final_result = cross_validate(model, X, y, cv=5, return_train_score=True)

        train_score = final_result['train_score'].mean()
        test_score = final_result['test_score'].mean()

        assert train_score > 0.89
        assert test_score > 0.78

        logger.info(f'Train Score: {train_score}')
        logger.info(f'Test Score: {test_score}')

        logger.info('Updating model...')
        update_model(model)

        logger.info('Generating model report...')
        validation_score = model.score(X_test, y_test)
        save_simple_metrics_report(train_score, test_score, validation_score, model)

        y_pred = model.predict(X_test)
        get_model_performance(y_test, y_pred)

        logger.info('Training Finished')

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")

    except Exception as e:
        logger.exception("An error occurred during training.")

if __name__ == "__main__":
    main()