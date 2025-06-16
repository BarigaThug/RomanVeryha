import matplotlib.pyplot as plt

# Przykładowe dane dokładności z epok (dla ilustracji)
epochs = list(range(1, 11))
accuracy = [0.61, 0.68, 0.72, 0.75, 0.78, 0.80, 0.81, 0.82, 0.825, 0.83]

# Tworzenie wykresu
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracy, marker='o', linestyle='-', color='green')
plt.title('Dokładność modelu GRU w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Dokładność (Accuracy)')
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()
