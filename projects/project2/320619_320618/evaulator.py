from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

class Evaluator:
    def __init__(self, score_metric):
        self.score_metric = score_metric
        self.supported_metrics = {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'precision': precision_score,
            'recall': recall_score
        }

        if self.score_metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric. Choose from {list(self.supported_metrics.keys())}.")

    def evaluate(self, model, X, y, metric=None):
        if metric is None:
            metric = self.score_metric
        if metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric. Choose from {list(self.supported_metrics.keys())}.")

        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") and metric in ['roc_auc', 'log_loss'] else None
        metric_function = self.supported_metrics[metric]

        if metric in ['roc_auc', 'log_loss'] and y_pred_proba is not None:
            return metric_function(y, y_pred_proba)
        else:
            return metric_function(y, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title("ROC-AUC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def generate_report(self, model, scores, X, y):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        report = classification_report(y, y_pred, output_dict=True)

        print("\nClassification Report:")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"Class {label}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")

        self.plot_confusion_matrix(y, y_pred)

        if y_pred_proba is not None:
            self.plot_roc_curve(y, y_pred_proba)

        print(scores)

