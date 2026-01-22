import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('final_normal_rw100.csv')
plt.figure(1)
plt.plot(df['z_score'][:200])
plt.figure(2)

plt.plot(df['p1'][:200])

plt.figure(3)
plt.plot(df['p2'][:200])
plt.show()