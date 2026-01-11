import matplotlib.pyplot as plt

# comme mon google colab a atteint la limite de l'utilisation de GPU, j'ai copié les résultats ici pour faire des visualisations intégrant dans le rapport
steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

train_loss = [
    3.316300, 2.983200, 2.649400, 2.544100, 2.231000, 2.113100, 2.074300,
    1.967100, 1.880400, 1.720300, 1.690000, 1.650800, 1.562100, 1.549800
]

val_loss = [
    2.631971, 2.206193, 1.982588, 1.866934, 1.751519, 1.728626, 1.623935,
    1.601709, 1.540167, 1.528521, 1.482500, 1.468524, 1.429536, 1.432408
]

cer = [
    0.515768, 0.474887, 0.463875, 0.429001, 0.421659, 0.425330, 0.392124,
    0.396963, 0.391290, 0.376272, 0.370432, 0.367929, 0.363090, 0.361755
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Loss
ax1.plot(steps, train_loss, label='Training Loss', color='blue', marker='o', linestyle='-', linewidth=2, markersize=4)
ax1.plot(steps, val_loss, label='Validation Loss', color='orange', marker='s', linestyle='--', linewidth=2, markersize=4)

ax1.set_title('Dynamique de la perte (Loss)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Steps (Pas d\'entraînement)', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

# CER
ax2.plot(steps, cer, label='CER (Character Error Rate)', color='#2ca02c', marker='^', linestyle='-', linewidth=2, markersize=5)

min_cer = min(cer)
min_step = steps[cer.index(min_cer)]
ax2.annotate(f'Min: {min_cer:.3f}', xy=(min_step, min_cer), xytext=(min_step-200, min_cer+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

ax2.set_title('Évolution du CER (Taux d\'erreur)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Steps (Pas d\'entraînement)', fontsize=12)
ax2.set_ylabel('CER', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)