# Wstęp

Celem jest przeanalizowanie tunowalności hiperparametrów 3 wybranych algorytmów uczenia maszynowego (np. xgboost, random forest, elastic net) na co najmniej 4 zbiorach danych. Do tunowania modeli należy wykorzystać min. 2 różne techniki losowania punktów (opisane dokładniej poniżej).

### Metody samplingu

1.  Co najmniej jedna metoda powinna się opierać na wyborze punktów z rozkładu jednostajnego. Przykładami mogą być:

-   Uniform grid search
-   Random Search

  **Uwaga: dla wszystkich zbiorów danych w tym kroku powinniśmy korzystać z tej samej ustalonej siatki hiperparametrów dla każdego algorytmu.**

2.  Co najmniej jedna technika powinna opierać się na technice bayesowskiej

-   Bayes Optimization
    
    _warto wykorzystać pakiet SMAC3 do dostosowania metody, ale może być też scikit-optimize i funkcja BayesSearchCV_
    

Wyniki z poszczególnych metod tunowania (historia tuningu) powinny być wykorzystywane do wyznaczenia tunowalności algorytmów.

Tunowalność algorytmów i hiperparametrów powinna być określona zgodnie z definicjami w [Tunability: Importance of Hyperparameters of Machine Learning Algorithms](https://jmlr.org/papers/volume20/18-444/18-444.pdf).



### Punkty, które należy rozważyć

Na podstawie wyników zgromadzonych w eksperymencie opisanym w sekcji [Wstęp] (#wstep) należy opisać i przeanalizować wyniki pod kątem: 

1.  ile iteracji każdej metody potrzebujemy żeby uzyskać stabilne wyniki optymalizacji
    
2.  określenie zakresów hiperparametrów dla poszczególnych modeli - motywacja wynikająca z literatury
    
3.  tunowalność poszczególnych algorytmów 

*lub* 

4. tunowalność poszczególnych hiperparametrów
        
5.  czy technika losowania punktów wpływa na różnice we wnioskach w punktach 3. i 4. dotyczących tunowalności algorytmów i hiperparametrów - Odpowiedź na pytanie czy występuje bias sampling.
    

### Potencjalne punkty rozszerzające PD

-   Zastosowanie testów statystycznych _do porównania różnic wyników pomiędzy technikami losowania hiperparametrów_
-   Zastosowanie **[Critical Difference Diagrams](https://github.com/hfawaz/cd-diagram#critical-difference-diagrams) -** w przypadku zastosowania większej liczby technik losowania punktów
-   Zaproponowanie wizualizacji i analiz wyników innych niż użyte w cytowanym artykule

### Oczekiwany wynik

Na przygotowanie rozwiązania projektu będą składały się następujące elementy:

-   raport opisujący wykorzystane metody i wyniki eksperymentów dla obu technik (maksymalnie 4 strony A4),
-   prezentacja podsumowująca rozwiązanie
  
### Oddanie projektu

Wszystkie punkty z sekcji _Szczegóły rozwiązania_ należy umieścić w katalogu o nazwie `NUMERINDEKSU1_NUMERINDEKSU2` lub `NUMERINDEKSU1_NUMERINDEKSU2_NUMERINDEKSU3`. Tak przygotowany **katalog należy umieścić na repozytorium przedmiotu w folderze `projects/project1`.**

### Zajęcia projektowe i konsultacje

<div class="tg-wrap"><table style="undefined;table-layout: fixed; width: 562px"><colgroup>
<col style="width: 157.2px">
<col style="width: 405.2px">
</colgroup>
<tbody>
  <tr>
    <td>Data</td>
    <td>Temat</td>
  </tr>
  <tr>
    <td>09.10.2024<br></td>
    <td>Intro to P1</td>
  </tr>
  <tr>
    <td>16.10.2024</td>
    <td>Konsultacje dotyczące danych</td>
  </tr>
  <tr>
    <td>23.10.2024</td>
    <td>Konsultacje dotyczące optymalizacji GS, RS, BO</td>
  </tr>
  <tr>
    <td>30.10.2024<br></td>
    <td>Konsultacje dotyczące optymalizacji GS, RS, BO</td>
  </tr>
  <tr>
    <td>06.11.2024</td>
    <td>Konsultacje dotyczące analizy wyników</td>
  </tr>
  <tr>
    <td>13.11.2024</td>
    <td>Konsultacje dotyczące analizy wyników</td>
  </tr>
  <tr>
    <td>20.11.2024</td>
    <td>Prezentacje P1</td>
  </tr>
</tbody>
</table></div>


### Terminy 

Termin oddania pracy domowej to **15.11.2024 EOD**.
Prezentacje będą się odbywały na zajęciach projektowych w dniu **20.11.2024**.
