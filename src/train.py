import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

def train_model():
  df = pd.read_csv("data/otodom.csv")
  print(len(df))
  df = df[df['price'].notnull()]
  X = df[['area']]
  y = df['price']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  model = RandomForestClassifier(max_depth=5, n_estimators=5)
  model.fit(X_train, y_train)

  dump(model, 'models/model.pkl')

if __name__ == '__main__':
  train_model()