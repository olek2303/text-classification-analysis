# Analiza danych tekstowych i klasyfikacja sentymentu

Ten projekt dotyczy analizy danych tekstowych oraz porównania różnych technik przetwarzania tekstu w kontekście klasyfikacji sentymentu. Dane pochodzą z Twittera i dotyczą COVID-19. W projekcie zastosowano różne techniki przetwarzania tekstu oraz modele klasyfikacyjne, aby ocenić ich skuteczność.

## Struktura notebooka

### 1. Wczytanie danych
- Dane treningowe i testowe są wczytywane z plików CSV: `Corona_NLP_train.csv` i `Corona_NLP_test.csv`.
- Dane są łączone w jeden zbiór, a niepotrzebne kolumny (`UserName`, `ScreenName`, `Location`, `TweetAt`) są usuwane.

### 2. Eksploracja danych
- Analiza rozkładu sentymentów w danych (`Sentiment`) oraz utworzenie uproszczonych klas (`CondensedSentiment`), gdzie:
  - `Extremely Positive` i `Positive` są łączone w `Positive`.
  - `Extremely Negative` i `Negative` są łączone w `Negative`.

### 3. Przetwarzanie tekstu
- **Sprowadzenie tekstu do małych liter**: Wszystkie znaki w tweetach są zamieniane na małe litery.
- **Usunięcie oznaczeń użytkowników i linków**: Usuwane są wzmianki zaczynające się od `@` oraz linki zaczynające się od `http`.
- **Usunięcie słów stopu**: Słowa stopu są usuwane przy użyciu biblioteki NLTK.
- **Usunięcie znaków interpunkcyjnych i cyfr**: Usuwane są wszystkie znaki interpunkcyjne i cyfry.
- **Lematyzacja**: Słowa są sprowadzane do ich podstawowych form przy użyciu bibliotek NLTK i SpaCy.
- **Stemming**: Słowa są sprowadzane do ich rdzeni przy użyciu algorytmu PorterStemmer.

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
- Wyniki metryk są zapisywane do plików JSON (`eval_metrics.json`, `eval_metrics_BERT.json`) oraz CSV (`accuracy_results.csv`).
- Wyniki są prezentowane w formie tabelarycznej.

### 8. Omówienie wyników
- Najlepsze wyniki uzyskano dla modelu BERT trenowanego na danych bez modyfikacji (oryginalny tekst).
- Wyniki innych modeli są porównywane w kontekście różnych technik przetwarzania tekstu.
- Poniżej przedstawiona tabela Accuracy uzyskanych wyników:

| Dataset   | Naive Bayes |   SVM    | Random Forest | Logistic Regression |   BERT   |
|-----------|-------------|----------|----------------|----------------------|----------|
| NLTK      |   0.673992  | 0.781119 |     0.724887   |       0.793220       | 0.820467 |
| SpaCy     |   0.670878  | 0.785123 |     0.730848   |       0.791618       | 0.814238 |
| Stemmed   |   0.668921  | 0.781475 |     0.735208   |       0.788059       | 0.767297 |
| Processed |   0.673192  | 0.783789 |     0.719726   |       0.790551       | 0.824027 |
| Original  |   0.676662  | 0.795089 |     0.692410   |       0.802651       | 0.858954 |


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
