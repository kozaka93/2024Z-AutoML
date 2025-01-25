import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance



def data_overview(df):
    print("Liczba wierszy:", df.shape[0])
    print("Liczba kolumn:", df.shape[1])
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print("Liczba zmiennych kategorycznych:", len(categorical_columns))
    print("Liczba zmiennych numerycznych:", df.shape[1]-len(categorical_columns))
    if total_missing > 0:
        print("Braki danych:")
        print(missing_values[missing_values > 0])
        print(f"Całkowita liczba brakujących wartości we wszystkich kolumnach: {total_missing}")
    else:
        print("Nie ma braków danych.")



def summarize_selected_features(selected_features):
    num_selected_features = len(selected_features)
    #random_sample = np.random.choice(selected_features, size=min(7, num_selected_features), replace=False)
    
    print(f"Łącznie wybrano {num_selected_features} cech.")
    print(f"Wybrane cechy:")
    print(selected_features)


def plot_confusion_matrix(confusion):
    labels=["jadalny","trujący"]
    plt.figure(figsize=(8, 6))
    cmap = sns.light_palette("#8B4513", as_cmap=True)  
    ax = sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'liczba'})
    
    bottom_left_x = 0.25  
    bottom_left_y = 1.75  
    ax.text(bottom_left_x, bottom_left_y, '⚠', fontsize=20, color='red', ha='center', va='center', fontweight='bold')

    plt.xlabel('Przewidywane')
    plt.ylabel('Prawdziwe')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_auc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#8B4513', label=f'Krzywa ROC (AUC = {roc_auc:.2f})')  
    plt.plot([0, 1], [0, 1], linestyle='--', color='#D2B48C') 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Krzywa')
    plt.legend()
    plt.show()


def plot_metrics_bar(metrics,weighted_score):
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics[m] for m in metric_names]
    values.append(weighted_score)
    metric_names.append('weighted_score')

    colors = ['#D2B48C', '#D2B48C', '#D2B48C', '#D2B48C', '#8B4513']  
    metric_names_2=['Dokładność', 'Precyzja', 'Czułość', 'F1', 'Custom score']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_names_2, y=values, palette=colors)
    plt.xlabel('metryki')
    plt.ylabel('wyniki')
    plt.title('Metryki modelu')
    plt.ylim(0, 1.1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    plt.show()



def generate_model_analysis_from_metrics(metrics):
    conclusions = []
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']

    # 1. Accuracy
    if accuracy > 0.90:
        conclusions.append(
            "Model osiągnął bardzo wysoką dokładność (>90%).\n"
            "To oznacza, że jest wyjątkowo skuteczny w klasyfikacji grzybów jako jadalne lub trujące."
        )
    elif accuracy > 0.80:
        conclusions.append(
            "Model ma dobrą dokładność (>80%).\n"
            "Wyniki są zadowalające, ale można je poprawić dla jeszcze lepszej ochrony grzybiarzy."
        )
    else:
        conclusions.append(
            "Model osiągnął niską dokładność (<80%).\n"
            "Może być ryzykowny w praktycznym zastosowaniu i wymaga optymalizacji."
        )

    # 2. Sensitivity (Recall)
    if recall > 0.90:
        conclusions.append(
            "Model ma bardzo wysoką czułość (>90%),\n"
            "co oznacza, że potrafi niemal bezbłędnie wykrywać trujące grzyby.\n"
            "To kluczowa cecha dla zapewnienia bezpieczeństwa."
        )
    elif recall > 0.80:
        conclusions.append(
            "Model ma dobrą czułość (>80%),\n"
            "ale istnieje ryzyko, że niektóre trujące grzyby mogą zostać sklasyfikowane jako jadalne."
        )
    else:
        conclusions.append(
            "Czułość modelu jest niska (<80%).\n"
            "Model może pomijać zbyt wiele trujących grzybów, co stanowi poważne zagrożenie."
        )

    # 3. Precision
    if precision > 0.90:
        conclusions.append(
            "Model ma bardzo wysoką precyzję (>90%),\n"
            "co oznacza, że większość grzybów sklasyfikowanych jako trujące faktycznie jest trująca."
        )
    elif precision > 0.80:
        conclusions.append(
            "Model ma dobrą precyzję (>80%),\n"
            "ale może zdarzyć się, że niektóre jadalne grzyby zostaną niesłusznie oznaczone jako trujące."
        )
    else:
        conclusions.append(
            "Precyzja modelu jest niska (<80%).\n"
            "Może prowadzić do błędów, gdzie jadalne grzyby są mylone z trującymi,\n"
            "co może zniechęcać użytkowników."
        )

    # 4. F1-score
    if f1 > 0.90:
        conclusions.append(
            "Model osiągnął bardzo wysoki wynik F1 (>90%),\n"
            "co oznacza, że dobrze równoważy precyzję i czułość."
        )
    elif f1 > 0.80:
        conclusions.append(
            "Model ma dobry wynik F1 (>80%),\n"
            "co wskazuje na solidną równowagę między precyzją i czułością."
        )
    else:
        conclusions.append(
            "Wynik F1 modelu jest niski (<80%).\n"
            "Model może mieć trudności z osiągnięciem dobrej równowagi między precyzją i czułością."
        )

    # 5. Waga metryk
    conclusions.append(
        "W przypadku klasyfikacji grzybów kluczowe znaczenie ma czułość (sensitivity/recall),\n"
        "ponieważ pomyłka w postaci zaklasyfikowania trującego grzyba jako jadalny\n"
        "może prowadzić do poważnych konsekwencji zdrowotnych.\n"
        "Dlatego model powinien być zoptymalizowany pod kątem minimalizacji tego ryzyka."
    )


    for conclusion in conclusions:
        print(conclusion)
        print() 


def plot_mushroom_balance(y_train):
    # Liczba elementów w każdej klasie
    class_counts = np.bincount(y_train)
    total = sum(class_counts)
    proportions = class_counts / total
    labels=["jadalny","trujący"]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Rysowanie "kapelusza" z podziałem na klasy
    r = 1  
    cap_start = 0
    colors = ['#8B4513', '#D2B48C']
    for i, proportion in enumerate(proportions):
        cap_end = cap_start + proportion * np.pi  
        theta_segment = np.linspace(cap_start, cap_end, 100)
        x_segment = np.concatenate(([0], r * np.cos(theta_segment), [0]))  
        y_segment = np.concatenate(([0], r * np.sin(theta_segment), [0]))  
        ax.fill(x_segment, y_segment, color=colors[i], label=f'{labels[i]}: {proportion:.1%}')
        

        mid_angle = (cap_start + cap_end) / 2
        x_text = 1.1 * np.cos(mid_angle)
        y_text = 1.1 * np.sin(mid_angle)
        ax.text(x_text, y_text, f"{proportion:.1%}", ha='center', va='center', fontsize=12, color='black')
        
 
        ax.plot([0, np.cos(cap_end)], [0, np.sin(cap_end)], color='black', linestyle='-', linewidth=1.5)
        
        cap_start = cap_end  
    
  
    stem_top = -0.02 
    stem_bottom = -0.3  
    x_stem = [-0.1, 0.1, 0.1, -0.1] 
    y_stem = [stem_top, stem_top, stem_bottom, stem_bottom]
    ax.fill_betweenx(y_stem, x_stem[0], x_stem[1], color='white', edgecolor='black')
    

    ax.axis('equal')
    ax.axis('off')
    ax.set_ylim(-0.35, 1.1)  
    plt.legend(loc='upper right', fontsize=10, frameon=False)  
    plt.title('Zbalansowanie danych')
    plt.show()



    max_proportion = proportions.max()
    if max_proportion > 0.9:
        print("Zbiór jest wyraźnie niezbalansowany.")
    elif max_proportion > 0.8:
        print("Zbiór jest niezbalansowany.")
    elif max_proportion>0.65:
        print("Zbiór jest lekko niezbalansowany.")

    else:
        print("Zbiór jest zbalansowany.")



