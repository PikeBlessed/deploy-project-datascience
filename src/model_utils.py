from sklearn.linear_model import Ridge
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def update_model(model: Ridge) -> None:
    dump(model, 'model/ridge_model.pkl')

def save_simple_metrics_report(train_score:float, test_score:float, validation_score:float, model) -> None:
    with open('report.txt', 'w') as report_file:
        report_file.write(f'# Model Pipeline Description'+'\n')

        report_file.write(f'### Train Score: {train_score}'+'\n')
        report_file.write(f'### Test Score: {test_score}'+'\n')
        report_file.write(f'### Validation Score: {validation_score}'+'\n')

def get_model_performance(y_real: pd.Series, y_pred: pd.Series) -> None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax=ax)
    ax.set_xlabel('Predicted Reach Content')
    ax.set_ylabel('Real Reach Content')
    ax.set_title('R2 of Model Prediction')
    fig.savefig('model_prediction.png')