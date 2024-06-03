import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = np.loadtxt('D:\project\DNApred_ResNet\data\motifs\\6mA_S.cerevisiae-dreme\GAAAAARA_1.3e-7.txt', dtype=int)
print(data)
columns = ['A', 'C', 'G', 'T']
df = pd.DataFrame(data, columns=columns)
df = logomaker.Logo(df)
ax = df.ax
ax.axis('off')
plt.savefig('D:\project\DNApred_ResNet\data\motifs\\6mA_S.cerevisiae-dreme\GAAAAARA_1.3e-7.png')
plt.show()

