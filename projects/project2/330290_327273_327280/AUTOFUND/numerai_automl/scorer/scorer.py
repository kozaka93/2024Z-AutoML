# numerai_automl/scorer.py

from typing import Dict
import pandas as pd
from numerai_tools.scoring import numerai_corr, correlation_contribution


class Scorer:
    def __init__(self):
        pass

    def compute_scores(self, data: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """
        Computes various scoring metrics for predictions.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing the following columns:
            - 'era': The era of the data.
            - 'predictions': The predicted values.
            - 'target': The actual target values.
        - target_name (str): The name of the target column.

        Returns:
        - pd.DataFrame: A DataFrame with the following columns:
            - 'mean': The mean of the predictions.
            - 'std': The standard deviation of the predictions.
            - 'sharpe': The Sharpe ratio of the predictions.
            - 'max_drawdown': The maximum drawdown of the predictions.

        The DataFrame will have different rows for each set of predictions, 
        with metrics calculated for each.
        """

        prediction_cols = [col for col in data.columns if "prediction" in col]

        correlations = data.groupby("era").apply(
            lambda d: numerai_corr(d[prediction_cols], d[target_name])
        )
        cumsum_corrs = correlations.cumsum()

       
        target_summary_metrics = {}

        for pred_col in prediction_cols:
            target_summary_metrics[pred_col] = self._get_summary_metrics(
                correlations[pred_col], cumsum_corrs[pred_col]
            )
        pd.set_option('display.float_format', lambda x: '%f' % x)
        summary = pd.DataFrame(target_summary_metrics).T
        summary.index.name = 'predictions' 

        return summary

    def compute_cumsum_correlation_per_era(self, data: pd.DataFrame, target_name: str = 'target') -> pd.DataFrame:
        """
        Computes the cumulative sum of correlations per era for each model.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing the following columns:
            - 'era': The era of the data.
            - 'predictions': The predicted values.
            - 'target': The actual target values.
        - target_name (str): The name of the target column.

        Returns:
        - pd.DataFrame: A DataFrame with the cumulative sum of correlations per era for each model.
        """

        prediction_cols = [col for col in data.columns if "prediction" in col]

        correlations = data.groupby("era").apply(
            lambda d: numerai_corr(d[prediction_cols], d[target_name])
        )
        cumsum_corrs = correlations.cumsum()

        return cumsum_corrs

    def get_mmc(self, data: pd.DataFrame, meta_model_col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes per-era MMC between our predictions, the meta-model and the target.
        Parameters:
        - data (pd.DataFrame): A DataFrame containing the following columns:
            - 'era': The era of the data, we will group by this.
            - 'predictions': The predicted values for each model - these should be named 'prediction_model_name'.
            - 'target': The actual target values.
            - meta_model_col (str): The column name of the meta-model predictions.
        Returns:
            - per_era_mmc (pd.DataFrame): A DataFrame with the MMC between our predictions, the meta-model
            and the target per era.
            - cumsum_mmc (pd.DataFrame): A DataFrame with the cumulative sum of MMC between our predictions,
            the meta-model and the target per era.
            - summary (pd.DataFrame): A DataFrame with the summary statistics of the MMC between our predictions,
            the meta-model and the target: mean, std, sharpe, max_drawdown.
        """
        prediction_cols = [col for col in data.columns if "prediction" in col]
        per_ear_mmc = data.groupby('era').apply(
            lambda x: correlation_contribution(
                x[prediction_cols], x[meta_model_col], x["target"]
            )
        )
        cumsum_mmc = per_ear_mmc.cumsum()
        summary = pd.DataFrame(self._get_summary_metrics(per_ear_mmc, cumsum_mmc))
        return per_ear_mmc, cumsum_mmc, summary

    @staticmethod
    def _get_summary_metrics(scores, cumsum_scores):
        summary_metrics = {}
        # per era correlation between predictions of the model trained on this target and cyrus
        mean = scores.mean()
        std = scores.std()
        sharpe = mean / std
        rolling_max = cumsum_scores.expanding(min_periods=1).max()
        max_drawdown = (rolling_max - cumsum_scores).max()
        return {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
