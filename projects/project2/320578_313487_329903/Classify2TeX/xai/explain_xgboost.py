import shap
import matplotlib.pyplot as plt
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import cm
import os


class ExplainXGBoost:
    '''
    This class contains methods to explain the model for XGBoost model.
    '''
    def __init__(self, model):
        self.model = model

    def save_global_feature_importance_shap(self, data, dataset_name):
        '''
        This method returns the global feature importance using SHAP library and plots the bar graph.
        It takes the data on which the importance is calculated.
        '''
        self.dataset_name = dataset_name    
        # If the data is too large, then take random 1000 samples, because SHAP explainer is computationally expensive.
        if len(data) > 1000:
            data_sample = data.sample(1000)
        else:
            data_sample = data.copy()
            
        explainer = shap.Explainer(self.model, data_sample)

        shap_values = explainer(data_sample)
        
        # Create SHAP bar plot
        shap_plot = shap.plots.bar(shap_values, max_display=15, show=False)

        # Save the plot
        os.makedirs(f'Results/{self.dataset_name}/XAI/XGBoost', exist_ok=True)
        plt.savefig(f'Results/{self.dataset_name}/XAI/XGBoost/global_feature_importance_shap.png', dpi=300, bbox_inches='tight')
        plt.close()

        return 
    
    def show_feature_importance_by_classes_shap(self, X, y):
        '''
        This method returns the feature importance by classes using SHAP library and plots the bar graph.
        It takes the data on which the importance is calculated.
        '''
        data_1 = X[y == 1]
        data_2 = X[y == 0]
        
        # if the data is too large, then take random 1000 samples, because SHAP explainer is computationally expensive.
        if len(data_1) > 1000:
            data_sample_1 = data_1.sample(1000)
        else:
            data_sample_1 = data_1.copy()

        if len(data_2) > 1000:
            data_sample_2 = data_2.sample(1000)
        else:
            data_sample_2 = data_2.copy()

        explainer = shap.Explainer(self.model, data_sample_1)

        shap_values_1 = explainer(data_sample_1)

        explainer = shap.Explainer(self.model, data_sample_2)
        
        shap_values_2 = explainer(data_sample_2)
         
        # change titles and display plots, change color
        print("Feature Importance for Class 1 using SHAP")
        shap.plots.bar(shap_values_1, max_display=10)

        print("Feature Importance for Class 0 using SHAP")
        shap.plots.bar(shap_values_2, max_display=10)

        plt.show()

        return
        
    def save_violin_summary_plot_shap(self, data, dataset_name):
        '''
        This method returns the summary plot using SHAP library and plots the violin plot.
        It takes the data on which the violin plot is plotted.
        '''
        self.dataset_name = dataset_name

        # If the data is too large, then take random 1000 samples, because SHAP explainer is computationally expensive.
        if len(data) > 1000:
            data_sample = data.sample(1000)
        else:
            data_sample = data.copy()

        explainer = shap.Explainer(self.model, data_sample)

        shap_values = explainer(data_sample)

        # Create SHAP violin plot
        shap_plot = shap.plots.violin(shap_values, max_display=15, show=False)

        # Save the plot
        os.makedirs(f'Results/{self.dataset_name}/XAI/XGBoost', exist_ok=True)
        plt.savefig(f'Results/{self.dataset_name}/XAI/XGBoost/violin_summary_plot_shap.png', dpi=300, bbox_inches='tight')
        plt.close()

        return
        
    def save_feature_importance_plot(self, dataset_name, max_features=15):
        '''
        Plots the feature importance using the model's `feature_importances_` attribute.

        Parameters:
        - max_features: Maximum number of top features to display (default: 15).
        '''
        self.dataset_name = dataset_name
        # get feature importances from the model
        importances = self.model.feature_importances_

        # get feature names
        feature_names = self.model.get_booster().feature_names

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # limit to top max_features
        feature_importance_df = feature_importance_df.head(max_features)

        # plot the feature importances
        plt.figure(figsize=(8, 4))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#7D50DD')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()

        # save the plot
        plt.savefig(f'Results/{self.dataset_name}/XAI/XGBoost/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        return



