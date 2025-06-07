# Analiza danych tekstowych i klasyfikacja sentymentu

Ten projekt dotyczy analizy danych tekstowych oraz porównania różnych technik przetwarzania tekstu w kontekście klasyfikacji sentymentu. Dane pochodzą z Twittera i dotyczą COVID-19. W projekcie zastosowano różne techniki przetwarzania tekstu oraz modele klasyfikacyjne, aby ocenić ich skuteczność.

## Zbiór Coronavirus tweets NLP - Text Classification

Analizowany zbiór danych z okresu pandemii koronawirusa posiada 45 000 rekordów. Ma zdefiniowanych 5 różnych oznaczeń 
tweetów (Extremely Positive, Positive, Neutral, Negative, Extremely Negative). W celu wyrównania rozkładu tych oznaczeń 
połączono kategorie Extremely Positive i Positive ze sobą. Analogicznie postąpiono z Negative i Extremely Negative. 
W wyniku tych działań, w zbiorze zostały 3 etykiety - Positive, Neutral, Negative. W zbiorze oprócz klasycznego 
preprocessowania danych tekstowych (tj. usunięcie słów stopu, 
znaków specjalnych, cyfr, itp.) zostały również usunięte takie części tweetów jak nazwy innych użytkowników czy linki.
Wykonano także porównanie bibliotek SpaCy oraz NLTK pod kątem różnic w zlematyzowanych zdaniach.
Okazało się także, że po preprocessowaniu niektóre tweety zostały całkowicie wyczyszczone. Dokładniej doszło 
do 19 takich przypadków. Ze względu na tak niedużą liczbę zdecydowaliśmy się usunąć te tweety, 
by nie wprowadzać fałszywych sygnałów podczas treningu.
Wyniki modeli można uznać za wystarczająco dobre, ponieważ plasują się one w okolicach 70%-80%.
Modele klasyczne (Naive Bayes, SVM, Random Forest, Logistic Regression) osiągały zbliżone wyniki niezależnie 
od zastosowanej techniki przetwarzania tekstu. Jednak najlepsze rezultaty uzyskano dla danych oryginalnych w 
przypadku BERT-a, czyli bez przeprowadzenia preprocessingu, a także bez nakładania na dane lematyzacji czy stemmingu. 
Ze względu na ograniczenia sprzętowe i czasowe, do trenowania modeli BERT zastosowano podpróbkowanie zbioru z zachowaniem 
proporcji klas, co pozwoliło na efektywne przeprowadzenie eksperymentów.

### Uzyskane wyniki accuracy dla różnych modeli i technik przetwarzania tekstu:
| Dataset   | Naive Bayes |   SVM    | Random Forest | Logistic Regression |   BERT   |
|-----------|-------------|----------|----------------|----------------------|----------|
| NLTK      |   0.673992  | 0.781119 |     0.724887   |       0.793220       | 0.820467 |
| SpaCy     |   0.670878  | 0.785123 |     0.730848   |       0.791618       | 0.814238 |
| Stemmed   |   0.668921  | 0.781475 |     0.735208   |       0.788059       | 0.767297 |
| Processed |   0.673192  | 0.783789 |     0.719726   |       0.790551       | 0.824027 |
| Original  |   0.676662  | 0.795089 |     0.692410   |       0.802651       | 0.858954 |


## Zbiór Spam Text Message Classification

Badany zbiór danych składa się z 5500 rekordów, jest to zatem niewielki zbiór. Zbiór posiada dwie kategorie - `spam` i `ham`. 
Można zaobserwować znaczną dysproporcję liczby rekordów oznaczonych jako `spam` w porównaniu do tych oznaczonych jako `ham`. 
Prawdopodobnie taka była specyfika zbieranych danych, a także samego zagadnienia, 
czyli wyraźniejsze odrzucanie e-maili czy wiadomości typu spam. Stąd też zdecydowaliśmy się nie modyfikować danych, 
a zostawić tę dysproporcję taką jaką zastaliśmy. Podczas preprocesowania zbioru danych natknęliśmy na słowa, 
które trudno byłoby przyrównać do istniejących słów w języku angielskim. Zdecydowaliśmy się na skorzystanie z algorytmu `textblob`, 
który miał za zadanie przywrócić, lub uratować błąd w zapisie słów. 
Zastosowanie `textblob` dało zamierzony efekt, ponieważ część zdań została ujednolicona. 
Mimo małego zbioru i sporej dyspropocji model uzyskał bardzo dobre wyniki - wszytskie modele uzyskały ponad 95% skuteczności. 
Stąd można wnioskować, że dane były bardzo dobrze wyselekcjonowane i modele łatwo odróżniały wiadomości typu `spam`/`ham`. 
Niezależnie od zastosowanej techniki przetwarzania tekstu i wybranego modelu, 
skuteczność klasyfikacji utrzymywała się na bardzo wysokim poziomie, co potwierdza, 
że problem klasyfikacji spam/ham na tym zbiorze jest stosunkowo łatwy do rozwiązania.

### Uzyskane wyniki accuracy dla różnych modeli i technik przetwarzania tekstu:
| Dataset   | Naive Bayes |   SVM    | Random Forest | Logistic Regression |   BERT   |
|-----------|-------------|----------|----------------|----------------------|----------|
| NLTK              |      0.966954 | 0.97342  |        0.972701 |              0.952586 | 0.962298 |
| SpaCy             |      0.96408  | 0.971983 |        0.970546 |              0.954023 | 0.960503 |
| Stemmed           |      0.965517 | 0.972701 |        0.971983 |              0.956897 | 0.960503 |
| Processed         |      0.967672 | 0.97342  |        0.974138 |              0.953305 | 0.962298 |
| ProcessedTextBlob |      0.967672 | 0.97342  |        0.970546 |              0.953305 | 0.962298 |
| Original          |      0.967672 | 0.979167 |        0.975575 |              0.963362 | 0.987433 |


## Zbiór Sentiment Analysis for Mental Health

Badany zbiór danych posiada 53000 rekordów. Kolumna status opisująca kategorię zdania - w tym przypadku o jakiej chorobie 
jest mowa w danym rzędzie - posiada 7 unikalnych wartości. Po wstępnym czyszczeniu i przetworzeniu danych (preprocessingu), 
dodano nowe kolumny zawierające przekształcone wersje zdań, które posłużyły w dalszej analizie. 
W procesie czyszczenia zdań ze zbędnych części, doszło do usunięcia nieznacznej liczby rekordów. 
Zaskakująca była duża liczba wystąpień nazw użytkowników i linków do innych stron internetowych. 
Celem eksperymentu było porównanie efektywności różnych technik przetwarzania 
języka naturalnego – w tym stemmingu i lematyzacji – w kontekście klasyfikacji za pomocą wybranych 
modeli uczenia maszynowego i sztucznej inteligencji. Pomimo nierównomiernego rozkładu klas w kolumnie status, 
uzyskano zadowalające wyniki predykcyjne. Jednym z możliwych podejść optymalizacyjnych mogłoby być połączenie czterech 
najmniej licznych klas w jedną kategorię, co skutkowałoby bardziej wyrównanym rozkładem danych i potencjalnie lepszymi wynikami modeli. 
Z drugiej strony – prowadziłoby to do utraty informacji o konkretnych jednostkach chorobowych. Alternatywnie, dla zachowania pełnej informacji, 
warto byłoby rozważyć zastosowanie technik augmentacji danych, które umożliwiłyby sztuczne powiększenie mniejszościowych klas. Warto zauważyć, 
że w 4 z 5 modoeli, najlepsze wyniki uzyskano na danych oryginalnych, bez zaawansowanego przetwarzania, co może sugerować 
utratę istotnych cech językowych podczas preprocessing'u. 
Z kolei model BERT, operujący na nieprzetworzonym tekście, osiągnął najwyższą skuteczność (~80%) spośród wszystkich testowanych metod.

### Uzyskane wyniki accuracy dla różnych modeli i technik przetwarzania tekstu:
| Dataset   | Naive Bayes |   SVM    | Random Forest | Logistic Regression |   BERT   |
|-----------|-------------|----------|----------------|----------------------|----------|
| NLTK              |      0.658842 |   0.763142 |        0.710043 |              0.753191 |   0.706968 |
| SpaCy             |      0.651854 |   0.757824 |        0.692267 |              0.747189 |   0.736488 |
| Stemmed           |      0.653373 |   0.764053 |        0.709359 |              0.754862 |   0.733449 |
| Processed         |      0.664616 |   0.768687 |        0.714448 |              0.7579   |   0.746473 |
| Original          |      0.669325 |   0.790565 |        0.706168 |              0.774537 |   0.805513 |


## Struktura notebooka

### 1. Wczytanie danych
- Dane treningowe i testowe są wczytywane z plików CSV: `Corona_NLP_train.csv` i `Corona_NLP_test.csv`.
- Dane są łączone w jeden zbiór, a niepotrzebne kolumny (`UserName`, `ScreenName`, `Location`, `TweetAt`) są usuwane.

### 2. Eksploracja danych
- Analiza rozkładu sentymentów w danych (`Sentiment`)
- Sprawdzenie braków

### 3. Przetwarzanie tekstu
- **Sprowadzenie tekstu do małych liter**: Wszystkie znaki w tweetach są zamieniane na małe litery.
- **Usunięcie oznaczeń użytkowników i linków**: Usuwane są wzmianki zaczynające się od `@` oraz linki zaczynające się od `http`.
- **Usunięcie słów stopu**: Słowa stopu są usuwane przy użyciu biblioteki NLTK.
- **Usunięcie znaków interpunkcyjnych i cyfr**: Usuwane są wszystkie znaki interpunkcyjne i cyfry.
- **Lematyzacja**: Słowa są sprowadzane do ich podstawowych form przy użyciu bibliotek NLTK i SpaCy.
- **Stemming**: Słowa są sprowadzane do ich rdzeni przy użyciu algorytmu PorterStemmer.
- **TextBlob**: Słowa są przetwarzane w ten sposób, by uzyskać ich prawidłową formę, gdy popełniono literówkę.

### 4. Przygotowanie danych do modeli
- Dane są wektoryzowane za pomocą `TfidfVectorizer` (maksymalnie 5000 cech).
- Dane są dzielone na zbiory treningowe i testowe dla różnych technik przetwarzania tekstu:
  - Lematyzacja (NLTK, SpaCy)
  - Stemming
  - Przetworzony tekst (bez lematyzacji/stemmingu)
  - Oryginalny tekst

### 5. Trening modeli klasyfikacyjnych
- Modele klasyfikacyjne użyte w projekcie:
  - **Naive Bayes**
  - **SVM**
  - **Random Forest**
  - **Logistic Regression**
  - **BERT** (Bidirectional Encoder Representations from Transformers)
- Dla każdego modelu obliczane są metryki:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - Czas treningu

### 6. Fine-tuning modelu BERT
- Model BERT jest trenowany na danych tekstowych z różnymi technikami przetwarzania.
- Dane są tokenizowane i przekształcane w odpowiedni format dla modelu BERT.
- Wyniki są oceniane na podstawie tych samych metryk co dla innych modeli.

### 7. Eksport wyników
- Wyniki metryk są zapisywane do plików JSON (`eval_metrics.json`) oraz CSV (`accuracy_results.csv`).
- Wyniki są prezentowane w formie tabeli.

## Wymagania
- Python 3.7+
- Biblioteki:
  - `pandas`
  - `matplotlib`
  - `nltk`
  - `spacy`
  - `scikit-learn`
  - `transformers`
  - `datasets`
  - `torch`
