# Data Mining Project
---

## Wybrany temat - System rekomendacyjny dla portalu multimedialnego


## Dataset:
- https://grouplens.org/datasets/movielens/

## Articles
- https://arxiv.org/html/2206.02631v2
- https://www.researchgate.net/publication/226098747_Content-based_Recommender_Systems_State_of_the_Art_and_Trends

## Potencjalne idee:
- reguły asocjacyjne
- knn

## Miary ewaluacyjne
Ranking z zakrytymi filmami: 
- normalized discounted cumulative gain (ndcg)
- mean reciprocal rank (mrr)
- average precision
- average first positive rank

Przygotować pipeline:
- normalizacja
- czyszczenie
- zbiory testowe/treningowe
- budowa modelu
- dopasowanie modelu do danych (fit)
- testowanie


# Do rozważenia
Obecnie mamy bias na filmy z dużą liczbą gatunków
latent semantic analysis
pca
user based

# Prezentacja (w powerpoint)
- zdefiniować problem
- określić zbiory danych
- zdefiniować miary ewaluacjne
- propozycje rozwiązania
- opisać proces trenowania (ile trwa itd)
- przedstawić wyniki ewaluacyjne
- wnioski