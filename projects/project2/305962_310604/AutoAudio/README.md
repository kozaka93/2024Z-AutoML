# Dokumentacja Auto Audio
## Wprowadzenie
Auto Audio to biblioteka Python służąca do automatycznej klasyfikacji dźwięku. Dostarcza elastyczne środowisko łączące ekstrakcję cech, trenowanie modeli i optymalizację hiperparametrów. Biblioteka wspiera zarówno tradycyjne modele uczenia maszynowego, jak i rozwiązania z zakresu deep learning.

### Do kogo skierowane jest Auto Audio?
Biblioteka Auto Audio skierowana jest przede wszystkim do entuzjastów uczenia maszynowego szukających praktycznych zastosowań w dziedzinie analizy i przetwarzania dźwięku. Ze względu na swoją prostotę i elastyczny, nieskomplikowany interfejs, biblioteka może być z powodzeniem wykorzystywana również przez osoby zaczynające swoją przygodę z dziedziną uczenia maszynowego. Ze względu na zastosowanie podejścia AutoML, biblioteka może być również używana przez badaczy w dziedzinie rozpoznawania dźwięku chcących zautomatyzować lub przyspieszyć proces uczenia i wyboru hiperparametrów dla tworzonych modeli. 

### Podobne biblioteki
Na rynku istnieją podobne rozwiązania, z których najbardziej znanym jest [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis). Warto jednak zaznaczyć, że nie jest to pakiet AutoML, ponieważ wymaga od użytkownika podejmowania decyzji, takich jak wybór modelu bazowego.
Mimo to pyAudioAnalysis stanowi najlepszy punkt odniesienia dla naszej biblioteki, ponieważ na rynku brakuje dedykowanych pakietów implementujących podejście AutoML w kontekście analizy dźwięku.   
Auto Audio wyróżnia się na tym tle automatyzacją procesu uczenia i doboru parametrów, co czyni go unikatowym rozwiązaniem w swojej kategorii.

### Instalacja 

Aby zainstalować bibliotekę wraz z jej głównymi zależnościami, użyj poniższego polecenia w katalogu, w którym znajduje się plik `setup.py`:

```bash
pip install -e . 
```
Powyższe polecenie zainstaluje bibliotekę w trybie edytowalnym uwzględniając następujące kluczowe zależności:

- pandas
- numpy
- scikit-learn
- librosa
- torch (dla modelu transformer)
- transformers (dla modelu transformer)
- skopt (dla optymalizacji bayesowskiej)

## Główne komponenty 

### AutoAudioModel
Główna klasa orkiestrująca cały proces klasyfikacji. Odpowiada za:

- Przetwarzanie wstępne danych
- Wybór i trenowanie modeli
- Optymalizację hiperparametrów
- Kontrolę czasu wykonania
- Ewaluację modeli

Konstruktor klasy `AutoAudioModel` przyjmuje tylko jeden parametr, `log`, który, jeśli zostanie ustawiony na wartość `False`, wyłączy logowanie.
Metoda `fit` przyjmuje następujące parametry:
- `data` – obiekt `pd.DataFrame`, który powinien zawierać dwie kolumny:
    - '`file_path`': pełna ścieżka do pliku audio.
    - '`label`': etykieta przypisana do pliku audio.
  
    Każdy wiersz reprezentuje pojedynczy plik audio oraz jego etykietę.
- `time_limit` – limit czasu na trening modelu, w sekundach.
- `tuner` – tuner używany do optymalizacji hiperparametrów.
- `random_state` – losowy seed dla modelu w celu zapewnienia powtarzalności wyników.

#### Podstawowe użycie: 
```python
from auto_audio_model import AutoAudioModel
from hyperparameter_tuner import HyperparameterTuner
import pandas as pd

# Przygotowanie danych
data = pd.DataFrame({
    "file_path": ["sciezka/do/audio1.wav", "sciezka/do/audio2.wav"],
    "label": ["klasa1", "klasa2"]
})

# Inicjalizacja modelu
model = AutoAudioModel()

# Trenowanie modelu
model.fit(data, time_limit=500)

# Wykonanie predykcji
predictions = model.predict(new_data)
```

### Preprocessing
Biblioteka wykorzystuje pakiet [librosa](https://github.com/librosa/librosa) do ekstrakcji następujących cech audio:

- MFCC (Mel-frequency cepstral coefficients)
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero crossing rate

Dla każdej cechy obliczana jest zarówno **średnia** jak i **odchylenie standardowe**, co pozwala na uzyskanie bogatej reprezentacji sygnału. Taka metoda przetwarzania jest standardem wykorzystywanym w analizie audio i przekłada się na wysoką skuteczność w zadaniach rozpoznawania i klasyfikacji dźwięku.

### Tuning hiperparametrów
Za dostrajanie hiperparametrów modeli odpowiedzialna jest klasa `HyperparameterTuner`, która wspiera dwie metody przeszukiwania przestrzeni parametrów:
- Przeszukiwanie losowe (Random Search)
- Optymalizacja Bayesowska (Bayesian Search)

#### Parametry:
- `search_method`: Metoda przeszukiwania przestrzeni hiperparametrów.  
Wybór między wartościami "random" a "bayes".
- `scoring`: Metryka do optymalizacji.  
np. "accuracy", "balanced_accuracy"`, "f1", "roc_auc".
- `cv`: Liczba foldów w walidacji krzyżowej.
- `n_iter`: Liczba kombinacji parametrów do sprawdzenia.
- `random_state`: Ziarno dla powtarzalności.


#### Przykładowe użycie:
```python
# Inicjalizacja tunera
simple_tuner = HyperparameterTuner(
    search_method="random"
    scoring="accuracy",
    cv=5,
    n_iter=30,
    random_state=42
)

# Inicjalizacja modelu
model = AutoAudioModel()

# Trenowanie modelu
model.fit(data, time_limit=500, tuner=simple_tuner)
```

### Zarządzanie czasem wykonywania
Biblioteka zawiera wbudowane funkcje zarządzania czasem:

- Określanie limitu czasu dla treningu modelu
- Automatyczny wybór modeli na podstawie dostępnego czasu
- Inteligentna alokacja czasu między różne modele
- Automatyczne wykluczenie modelu transformer przy zbyt ciasnych ograniczeniach czasowych

### Obsługa błędów
Biblioteka obsługuje różne przypadki błędów:

- Uszkodzone pliki audio
- Niewystarczający czas na trening
- Brak CUDA dla modelu transformer
- Nieprawidłowy format danych
- Brakujące wymagane kolumny

### Postprocessing

W trakcie działania klasy `AutoAudioModel`, postprocessing obejmuje dwie kluczowe funkcjonalności:

1. Logowanie do konsoli:

    Metoda `log()` zapisuje komunikaty o stanie działania modelu (np. rozpoczęcie treningu, osiągniętą dokładność, zatrzymanie procesu) na konsolę, jeśli flaga logging_enabled jest ustawiona na True.

2. Zapis informacji o modelach do słownika self.info:

    W słowniku `self.info` przechowywane są szczegółowe dane dotyczące przebiegu treningu i osiągniętej dokładności dla każdego modelu.
    Kluczowe informacje:
    - `self.info["model_accuracies"]`: - dokładności uzyskane przez poszczególne modele.
    - ` self.info["best_accuracy"]`: - najlepsza uzyskana dokładność w procesie treningu.

## Dostępne modele
W bibliotece zaimplementowane zostały następujące modele:

### 1. Maszyna Wektorów Nośnych (SVM)
Tradycyjny model ML odpowiedni dla danych wysokowymiarowych.

#### Zakresy Hiperparametrów:

- `C`: [0.1, 1, 10, 100]
- `kernel`: ["linear", "rbf", "poly", "sigmoid"]
- `gamma`: ["scale", "auto"]
- `degree`: [2, 3, 4, 5]

### 2. K Najbliższych Sąsiadów (KNN)
Skuteczny model o niewielkim stopniu złożoności oparty na metrykach odległości.

#### Zakresy Hiperparametrów:

- `n_neighbors`: [3, 5, 7, 9, 11]
- `weights`: ["uniform", "distance"]
- `metric`: ["euclidean", "manhattan", "minkowski"]

### 3. Gradient Boosting (XGBoost)
Metoda uczenia maszynowego oparta o drzewa decyzyjne, tworząca modele iteracyjnie, minimalizując błędy poprzednich za pomocą optymalizacji gradientowej.

#### Zakresy Hiperparametrów:

- `n_estimators`: [50, 100, 200, 300]
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 1.0]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

### 4. Transformer (wav2vec2)
Model głębokiego uczenia oparty na architekturze wav2vec2 opracowanej przez Facebooka. Dostępny tylko przy obecności CUDA.

#### Domyślna konfiguracja:

- `model`: facebook/wav2vec2-base
- `learning_rate`: 3e-5
- `batch_size`: 32
- `epochs`: 10
- `warmup_ratio`: 0.1

## Zalecane praktyki

#### Przygotowanie Danych:
- Zapewnienie spójnego formatu audio.
- Dostarczenie bezwzględnych ścieżek do plików.
- Używanie zbalansowanej dystrybucji klas w celu poprawy jakości miar oceny i zapobieganiu przeuczania się modeli na klasie dominującej.

#### Zarządzanie czasem treningu:
- Ustawienie realistycznych limitów czasu w oparciu o rozmiar zbioru danych.
- Rozważenie wykluczenia modelu transformer w celu uzyskania szybkich wyników.
- Ustawienie niskiej ilości kombinacji badanej podczas tuningu hiperparametrów lub całkowite zrezygnowanie z tuningu jeśli proces ten jest zbyt czasochłonny.

#### Zarządzanie zasobami:
- Monitorowanie użycia pamięci dla dużych zbiorów danych.
- Uwzględnienie dostępności GPU przy wyborze modelu transformer.
- Używanie odpowiednich rozmiarów batch dla posiadanego sprzętu.


## Dodatkowe pliki i przykłady użycia 
W katalogu `examples` znajdują się trzy notatniki Jupyter, które prezentują różne zastosowania i funkcjonalności paczki:

1. `audio_mnist.ipynb`

    Zawiera pełny przepływ wykorzystania paczki na zbiorze danych Audio MNIST. Znajduje się tam demonstracja załadowania danych, przeprowadzenia treningu modelu oraz przedstawienie jak dokonać predykcji.

2. `gtzan.ipynb`

    Podobnie jak w przypadku Audio MNIST, ten notatnik prezentuje pełny przepływ pracy na zbiorze danych GTZAN, który jest popularnym zbiorem do klasyfikacji gatunków muzycznych. Pokazuje, jak skonfigurować modele oraz przeprowadzić analizę wyników.

3. `comparison.ipynb`

    W pliku tym znajduje się porównanie działania stworzonej paczki z pyAudioAnalysis. Zawiera porównanie użycia obu bibliotek oraz wyniki testów, które pozwalają ocenić różnice w wydajności i dokładności pomiędzy obiema bibliotekami.