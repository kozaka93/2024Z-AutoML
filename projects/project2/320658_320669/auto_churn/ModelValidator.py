import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report


class ModelValidator:
    def __init__(self, df: pd.DataFrame, target: str, model):
        self.df = df
        self.target = target
        self.features = [col for col in df.columns if col != target]
        x_train, x_test, y_train, y_test = train_test_split(self.df[self.features], self.df[self.target],
                                                            stratify=self.df[self.target], test_size=0.3,
                                                            random_state=10)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.is_fitted = False

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train)
        self.is_fitted = True

    def get_roc_plot(self):
        """
        Function to calculate auroc scores
        and plot roc curve
        :param model: already trained model
        :param x_test: data frame for prediction\
        :param y_test: target vector
        """
        assert self.is_fitted == True

        r_probs = [0 for _ in range(len(self.y_test))]
        model_probs = self.model.predict_proba(self.x_test)
        model_probs = model_probs[:, 1]
        r_auc = roc_auc_score(self.y_test, r_probs)
        model_auc = roc_auc_score(self.y_test, model_probs)
        print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
        print('Model: AUROC = %.3f' % (model_auc))
        print('Model: Gini = %.3f' % (2 * model_auc - 1))
        r_fpr, r_tpr, _ = roc_curve(self.y_test, r_probs)
        model_fpr, model_tpr, _ = roc_curve(self.y_test, model_probs)
        plt.figure(figsize=(12, 8))
        plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
        plt.plot(model_fpr, model_tpr, marker='.', label='Model (AUROC = %0.3f)' % model_auc)
        plt.title('ROC Plot')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig("./figures/roc_curve.png", bbox_inches="tight")
        plt.show()
        return model_auc, 2 * model_auc - 1

    def get_classification_report(self):
        y_pred = self.model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred, output_dict=True))

    def get_model_summary(self):
        self.get_classification_report()
        auc, gini = self.get_roc_plot()

