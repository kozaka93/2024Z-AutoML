**Celem projektu** jest stworzenie prototypu pakietu AutoML, który będzie adresowany do określonej grupy odbiorców i skoncentrowany na rozwiązywaniu konkretnego typu problemów uczenia maszynowego. Pakiet ma wspierać użytkowników na każdym etapie procesu automatycznego modelowania ML, zapewniając funkcjonalność w zakresie przetwarzania danych, selekcji i optymalizacji modeli, a także ewaluacji wyników.

## **Zakres projektu**

1. **Przetwarzanie danych (Preprocessing)**:
    
    System ma zapewniać kompleksowe przetwarzanie danych, uwzględniające charakterystykę zbiorów wejściowych. Funkcje preprocessingowe obejmą np. obsługę brakujących danych, normalizację, ekstrakcję cech oraz transformacje specyficzne dla wybranego typu problemu. 
    
2. **Selekcja i optymalizacja modeli**:
    
    Pakiet będzie wspierał wybór odpowiednich algorytmów uczenia maszynowego, dostosowanych do specyfiki problemu, oraz automatyczną optymalizację hiperparametrów za pomocą metod takich jak grid search, random search lub zaawansowane podejścia, np. optymalizacja bayesowska.
    
3. **Ewaluacja i podsumowanie wyników**:
    
    System umożliwi ocenę wybranych modeli przy użyciu wskaźników dostosowanych do problemu (np. dokładność, F1-score, RMSE) oraz generowanie przejrzystych raportów podsumowujących, które będą zrozumiałe zarówno dla ekspertów, jak i osób mniej zaawansowanych technicznie.
    

## **Kluczowe wymagania projektowe**

- **Rozszerzone rozwiązanie:** Jeden z komponentów (np. preprocessing lub model selection) musi zawierać innowacyjne podejście, takie jak dynamiczne dostosowanie do danych, zaawansowana analiza cech czy zautomatyzowana interpretacja wyników modeli.
- **Interfejs użytkownika:** System ma być dostępny w formie pakietu w Python, uzupełnionego o tutorial w Jupyter Notebooku, wyjaśniający poszczególne komponenty oraz prezentujący interpretację wyników na rzeczywistych danych.

## **Wymagane elementy projektu (40 pkt):**

1. **Jupyter Notebook z tutorialem (30 pkt):**
    - Kompletny przewodnik po funkcjonalnościach pakietu.
    - Przykładowy przepływ pracy z danymi, wyborem modelu i ewaluacją wyników.
    - Wyjaśnienie komponentów oraz interpretacja uzyskanych rezultatów.
    
    Dokładny podział punktów znajduje się w [sekcji poniżej](#Kryteria-oceny-Jupyter-notebook) .
    
2. **Pakiet implementacyjny (10 pkt):**
    - W pełni funkcjonalny kod pakietu w Pythonie.
    - Starannie sformatowane wyniki, gotowe do analizy i raportowania.
3. **Prezentacja projektu (5 pkt):**
    - Zwięzłe podsumowanie założeń, komponentów technicznych i osiągniętych wyników.

### **Kryteria oceny Jupyter notebook**

1. **Specyfikacja podstawowa (15 pkt):**
    - Określenie grupy docelowej użytkowników (3 pkt).
    - Zdefiniowanie specjalizacji narzędzia, np. analiza predykcyjna, klasyfikacja, prognozowanie szeregów czasowych (4 pkt).
    - Przegląd istniejących rozwiązań podobnych w założeniach do zaproponowanego rozwiązania,  (8 pkt).
2. **Komponenty techniczne (15 pkt):**
    - Funkcje preprocessingowe (4 pkt).
    - Automatyzacja wyboru i optymalizacji modeli (4 pkt).
    - Funkcje postprocessingowe, w tym generowanie raportów (4 pkt).
    - Jeden z komponentów musi być szczególnie rozbudowany, uwzględniając innowacyjne podejście (dodatkowe 3 pkt).

## **Harmonogram realizacji**

- **Termin oddania projektu:** 15 stycznia 2025, do końca dnia.
