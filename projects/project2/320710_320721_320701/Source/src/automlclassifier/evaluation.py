from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationReports:
    """
    Klasa wspomagająca ewaluację modeli.
    """

    @staticmethod
    def evaluate(model, X_test, y_test):
        """
        Ewaluacja wyników i wizualizacja.
        """
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print("=== Classification Report ===")
        print(class_report)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        return {"accuracy": acc, "f1_score": f1, "classification_report": class_report}

    @staticmethod
    def save_html_report(report, path="report.html"):
        """
        Generowanie raportu HTML.
        """
        template = f"""
<html>
<head><title>Model Evaluation Report</title></head>
<body>
    <h1>Model Evaluation Report</h1>
    <p><b>Accuracy:</b> { report['accuracy'] }</p>
    <p><b>F1 Score:</b> { report['f1_score'] }</p>
    <h2>Classification Report</h2>
    <pre>{ report['classification_report'] }</pre>
</body>
</html>
        """
        with open(path, "w") as f:
            f.write(template)
