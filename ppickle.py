import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('penguins.csv')
# print(penguin_df.head())

# borrar filas con valores nulos
df.dropna(inplace=True)

# valores de salida output
y = df['sex']

# Valores de entrada features
X = df[['bill_length_mm', 'bill_depth_mm','flipper_length_mm', 
                       'body_mass_g','island', 'species']]
# Codificación 
X = pd.get_dummies(X)
y, uniques = pd.factorize(y)

print('Variables de salida')
print(y)
print('Variables de entrada')
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)
# Modelo árboles aleatorios 
logreg = LogisticRegression(random_state=15)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
score = accuracy_score(y_pred, y_test)
print('Prescición del modelo: {}'.format(score))
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(logreg, rf_pickle)
# open() crea dos archivos pickle
rf_pickle.close()
#  wb (write bytes): indica a Python que queremos escribir, no leer, en el archivo output_penguin.pickle
output_pickle = open('output_penguin.pickle', 'wb')
# pickle.dump() escribe los archivos de Python en output_penguin.pickle
pickle.dump(uniques, output_pickle)
# close()  cierra los archivos
output_pickle.close()