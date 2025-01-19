import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

def plot_statistics_per_class(statistics, labels_type):
    """
    Rysuje wykres przedstawiający statystyki dla każdej klasy.

    Args:
        labels (list): Lista nazw klas.
        statistics (dict): Słownik statystyk z kluczami 'precision', 'recall', 'f1', 'auc', zawierający listy wyników.
    """
    x = np.arange(len(labels_type))  # Pozycje na osi X

    width = 0.2  # Szerokość słupka
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysowanie słupków dla każdej metryki
    ax.bar(x - width * 1.5, statistics['precision'], width, label='Precision', color='skyblue')
    ax.bar(x - width * 0.5, statistics['recall'], width, label='Recall', color='lightgreen')
    ax.bar(x + width * 0.5, statistics['f1'], width, label='F1 Score', color='salmon')
    ax.bar(x + width * 1.5, statistics['auc'], width, label='AUC', color='orange')
    ax.bar(x + width * 1.5, statistics['accuracy'], width, label='AUC', color='red')

    # Ustawienia osi i etykiet
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Statistics per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_type, rotation=90, ha='right')
    ax.legend()

    # Dodanie wartości nad słupkami
    for i, label in enumerate(labels_type):
        ax.text(i - width * 1.5, statistics['precision'][i] + 0.02, f"{statistics['precision'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i - width * 0.5, statistics['recall'][i] + 0.02, f"{statistics['recall'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width * 0.5, statistics['f1'][i] + 0.02, f"{statistics['f1'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width * 1.5, statistics['auc'][i] + 0.02, f"{statistics['auc'][i]:.2f}", ha='center', fontsize=8)
        ax.text(i + width * 1.5, statistics['accuracy'][i] + 0.02, f"{statistics['accuracy'][i]:.2f}", ha='center', fontsize=8)
        
    plt.tight_layout()
    plt.show()


def compute_and_plot_statistics(y_test_flat, y_pred_flat, labels_type, print_stats=False, plot_stats=False):
        """
        Oblicza statystyki dla każdej klasy i tworzy wykres.
    
        Args:
            y_test_flat (np.ndarray): Spłaszczona macierz testowa (n_samples, n_classes).
            y_pred_flat (np.ndarray): Spłaszczona macierz przewidywań (n_samples, n_classes).
            labels_type (list): Lista nazw klas.
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        accuracy_scores = []

        if labels_type is None:
            labels_type = [ f"class_{i}" for i in range(y_test_flat.shape[1])]
            
        for i in range(len(labels_type)):
            y_test_ = y_test_flat[:, i]
            y_pred_ = y_pred_flat[:, i]
    
            precision = precision_score(y_test_, y_pred_, zero_division=0)
            recall = recall_score(y_test_, y_pred_, zero_division=0)
            f1 = f1_score(y_test_, y_pred_, zero_division=0)
            accuracy = accuracy_score(y_test_, y_pred_)
            
            if sum(y_test_) != 0:
                auc = roc_auc_score(y_test_, y_pred_)
            else:
                auc = 0  # Brak danych do obliczenia AUC
    
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_scores.append(auc)
            accuracy_scores.append(accuracy)
            
            if print_stats == True:
                print(f"Class {labels_type[i]} ({i})  statistics:")
                print("   Precision:", precision)
                print("   Recall:", recall)
                print("   F1 Score:", f1)
                print("   Accuracy:", accuracy)
                print(f"   AUC for class {i}:", auc)
    
        statistics = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'auc': auc_scores,
            'accuracy': accuracy_scores
        }
        
        if plot_stats == True:
            plot_statistics_per_class(statistics, labels_type)
            
        return statistics


def plot_data_raport(X, y, labels_type):
    
    if labels_type is None:
        labels_type = [ f"class_{i}" for i in range(y_test_flat.shape[1])]
            
    # Liczba jedynek dla każdej klasy
    ones_count = np.sum(y, axis=0)
    
    # Procentowy udział jedynek
    total_samples = y.shape[0]
    ones_percentage = (ones_count / total_samples) * 100
    n_classes = y.shape[1]
    
    # Tworzenie wykresów
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    
    # Wykres 1: Liczba jedynek
    axes[0].bar(range(n_classes), ones_count, color='skyblue')
    axes[0].set_title("Liczba jedynek dla każdej klasy")
    axes[0].set_xlabel("Klasy")
    axes[0].set_ylabel("Liczba jedynek")
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels([f"{labels_type[i]}" for i in range(n_classes)], rotation=90)
    for i, count in enumerate(ones_count):
        axes[0].text(i, count + 0.5, str(count), ha='center', fontsize=9)
    
    # Wykres 2: Procentowy udział jedynek
    axes[1].bar(range(n_classes), ones_percentage, color='salmon')
    axes[1].set_title("Procentowy udział jedynek dla każdej klasy")
    axes[1].set_xlabel("Klasy")
    axes[1].set_ylabel("Procentowy udział (%)")
    axes[1].set_xticks(range(n_classes))
    axes[1].set_xticklabels([f"{labels_type[i]}" for i in range(n_classes)], rotation=90)
    for i, percentage in enumerate(ones_percentage):
        axes[1].text(i, percentage + 0.5, f"{percentage:.2f}%", ha='center', fontsize=9)
    
    # Dopasowanie układu i wyświetlenie
    plt.tight_layout()
    plt.show()