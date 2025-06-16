import matplotlib.pyplot as plt

# Dane do wykresu
epochs = list(range(1, 11))
loss = [0.69, 0.63, 0.59, 0.55, 0.50, 0.46, 0.43, 0.41, 0.39, 0.37]

# Wykres
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, marker='o', linestyle='-', color='red')
plt.title('Wartość funkcji straty (loss) podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata (Loss)')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')  
plt.show()
