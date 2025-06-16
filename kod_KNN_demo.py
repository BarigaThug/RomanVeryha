import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

dane = pd.read_csv("steam_games.csv")
dane = dane[["user_id", "game", "playtime_hours"]].dropna()

macierz = dane.pivot_table(index="user_id", columns="game", values="playtime_hours", fill_value=0)

model_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=5)
model_knn.fit(macierz)

uzytkownik_id = np.random.choice(macierz.index)
dystanse, indeksy = model_knn.kneighbors([macierz.loc[uzytkownik_id]])

print(f"Podobni użytkownicy do: {uzytkownik_id}")
print(macierz.index[indeks[0]])

rekomendacje = macierz.iloc[indeksy[0]].mean(axis=0).sort_values(ascending=False)
print("Najczęściej grane gry:\n", rekomendacje.head(5))
