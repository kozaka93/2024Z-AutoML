from sklearn.metrics import confusion_matrix
import warnings
from supertree import SuperTree
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns
import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('nbAgg')

def distribution_plots(aid):
        #create a folder for the plots
    df = aid.df_before
    X = df.drop(columns=[aid.target_column])
    y = df[aid.target_column]
    path = aid.path
    if not os.path.exists(f"{path}/distribution_plots"):
        os.makedirs(f"{path}/distribution_plots")

    #generate count plots for each feature not using sns
    for col in X.columns:
        #if more that 10 unique values make a histogram
        if len(X[col].unique()) > 10:
            plt.figure()
            X[col].hist()
            plt.title(f'{col} distribution')
            plt.savefig(f"{path}/distribution_plots/{col}_hist.png")
            plt.clf()
        else:
            #make countplots but sort the values on the x axis
            plt.figure()
            X[col].value_counts().sort_index().plot(kind='bar')
            plt.title(f'{col} distribution')
            plt.savefig(f"{path}/distribution_plots/{col}_count.png")
            plt.clf()

    return None

def correlation_plot(aid):
    #create a folder for the plots
    path = aid.path
    X = aid.X
    y = aid.y
    if not os.path.exists(f"{path}/correlation_plots"):
        os.makedirs(f"{path}/correlation_plots")

    corr = X.corr()
    #plot the correlation matrix the scale should be from -1 to 1
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.savefig(f"{path}/correlation_plots/correlation_matrix.png")
    plt.clf()

    #make correlation with y plots
    for col in X.columns:
        plt.figure()
        sns.violinplot(x=y, y=X[col])
        plt.title(f'{col} correlation with y')
        plt.savefig(f"{path}/correlation_plots/{col}_correlation.png")
        plt.clf()
    return None

def make_confusion_matrix(aid):
    #create a folder for the plots
    path = aid.path
    X_test = aid.X_test
    y_test = aid.y_test
    if not os.path.exists(f"{path}/confusion_matrix"):
        os.makedirs(f"{path}/confusion_matrix")

    #generate confusion matrix for each model
    for model in aid.best_models:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', vmin = 0)
        plt.title(f'{model.__class__.__name__} confusion matrix')
        plt.savefig(f"{path}/confusion_matrix/{model.__class__.__name__}_confusion_matrix.png")
        plt.clf()
    return None


def shap_feature_importance_plot(aid):
    # Create a folder for the plots
    path = aid.path
    # Use a subset of the data
    X = aid.X.head(int(len(aid.X) * 0.03))  # Adjust as needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X.columns)
    if not os.path.exists(f"{path}/shap_feature_importance"):
        os.makedirs(f"{path}/shap_feature_importance")

    # Generate SHAP feature importance for each model
    for model in aid.best_models:
        if model.__class__.__name__ == "LogisticRegression":
            #get feature importance from the model
            importance_values = np.abs(model.coef_).mean(axis=0)
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance_values
            }).sort_values(by='Importance', ascending=False).head(10)

            # Plot manually
            feature_importance.plot(kind='bar', x='Feature', y='Importance', legend=False)
            plt.title(f'{model.__class__.__name__} Feature Importance')
            #add some space below plot not to cut the x labels
            plt.tight_layout()
            plt.savefig(f"{path}/shap_feature_importance/{model.__class__.__name__}_custom_feature_importance.png") #TODO: make sure the names of columns aren't cropped in images
            plt.clf()
            continue

        # Select appropriate SHAP explainer
        if model.__class__.__name__ in ["DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)

        shap_values = explainer(X)

        # Handle multi-class models
        if len(shap_values.values.shape) > 2:  # Multi-class case
            shap_values_to_plot = shap_values.values.mean(axis=-1)  # Average across classes
        else:
            shap_values_to_plot = shap_values.values

        # Calculate feature importance
        importance_values = np.abs(shap_values_to_plot).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=False).head(10)

        # Plot manually
        feature_importance.plot(kind='bar', x='Feature', y='Importance', legend=False)
        plt.title(f'{model.__class__.__name__} Shap Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{path}/shap_feature_importance/{model.__class__.__name__}_custom_feature_importance.png")
        plt.clf()
    return None

def generate_supertree_visualizations(medaid, output_dir="supertree_visualizations"):

    """
    Generate and save SuperTree visualizations for models in the Medaid object.
    Parameters:
    - medaid: The Medaid object containing best models, training data, and metadata.
    - output_dir: Directory where visualizations will be saved (default: "supertree_visualizations").
    Output:
    - Saves HTML visualizations for each model in the specified directory.
    """

    # Ensure output directory exists
    output_path = os.path.join(medaid.path, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Feature and target names
    try:
        feature_names = feature_names=medaid.X_train.columns.tolist()  # Feature names from training data
        print(feature_names)
        target_names = [str(label) for label in medaid.y_train.unique()]  # Target names from training data
        print(target_names)
    except AttributeError as e:
        print(f"Error extracting feature or target names: {e}")
        return

    # Loop through best models and create visualizations for supported ones
    for model in medaid.best_models:
        model_name = model.__class__.__name__
        print(f"Processing model: {model_name}")

        try:
            st = SuperTree(
                model=model,
                feature_data = medaid.X_train.reset_index(drop=True),
                target_data = medaid.y_train.reset_index(drop=True),
                feature_names=feature_names,
                target_names=target_names,
            )
            print(f"SuperTree visualization for {model_name} created.")
            # Save the HTML visualization
            html_path = os.path.join(output_path, f"{model_name}_tree.html")
            #save the tree to html
            # st.save_html(html_path)
            st.save_html(html_path)
            print(f"Saved SuperTree visualization for {model_name} at: {html_path}")

        except Exception as e:
            print(f"Skipping model {model_name}: {e}")

    print("SuperTree visualizations generation complete.")
    return None

def makeplots(aid):
    ' This function generates all the plots for the medaid object'
    distribution_plots(aid)
    correlation_plot(aid)
    make_confusion_matrix(aid)
    shap_feature_importance_plot(aid)
    generate_supertree_visualizations(aid)

    return None


"""
if __name__ == "__main__": #main was created for testing purposes
    from medaid.training.medaid import MedAId
    medaid = MedAId(dataset_path='../../data/binary/alzheimers_disease_data.csv', target_column='Diagnosis', metric="recall", search="random", n_iter=3)
    print(medaid.path)
    medaid.train()
    print("finished_training")
    makeplots(medaid)
    medaid.save()
"""


