from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_iris()

df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].apply(lambda x : data.target_names[x])

sns.pairplot(df,hue='species')
plt.show()