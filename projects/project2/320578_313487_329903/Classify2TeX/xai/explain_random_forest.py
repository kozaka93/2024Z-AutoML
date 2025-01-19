import shap
import matplotlib.pyplot as plt
import pandas as pd

shap.initjs()

class ExplainRandomForest:
    '''
    This class contains methods to explain the model for RandomForestClassifier.
    '''
    def __init__(self, model):
        self.model = model

    def plot_feature_importance(self, feature_names, max_features=10):
        '''
        Plots the feature importance using the model's `feature_importances_` attribute.

        Parameters:
        - max_features: Maximum number of top features to display (default: 10).
        - feature_names: List of feature names.
        '''
        # get feature importances from the model
        importances = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # limit to top max_features
        feature_importance_df = feature_importance_df.head(max_features)

        # plot the feature importances
        plt.figure(figsize=(8, 4))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#C70039')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

        return