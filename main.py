import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np


def main():
    ds = pd.read_csv('./data/6class.csv')
    df = ds
    df['Spectral Class'] = df['Spectral Class'].map({'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6})
    df[['Blue', 'White', 'Orange', 'Red', 'Yellow']] = 0
    df.loc[df['Star color'].str.contains("Blue", case=False), 'Blue'] = 1
    df.loc[df['Star color'].str.contains("Whit", case=False), 'White'] = 1
    df.loc[df['Star color'].str.contains("Red", case=False), 'Red'] = 1
    df.loc[df['Star color'].str.contains("Yellow", case=False), 'Yellow'] = 1
    df.loc[df['Star color'].str.contains("Orange", case=False), 'Orange'] = 1

    df.drop(labels=['Star color'], axis=1, inplace=True)

    cols = list(df.columns.values)
    cols.pop(cols.index('Star type'))
    df = df[cols + ['Star type']]

    df_n = df
    sample = np.array([df.iloc[6, :-1]])
    print(sample)
    samplet = StandardScaler().fit_transform(sample)
    print(samplet)
    df_n.iloc[:, :-1] = StandardScaler().fit_transform(df.iloc[:, :-1])

    X = df_n.iloc[:, :-1]
    y = df_n['Star type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)

    print(df_n.head())

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    predicted = model.predict([np.array(X_test)[0]])

    print(predicted)
    print(y_test)
    pass

if __name__ == '__main__':
    main()

