import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_data(df):

    X = df.drop(
        columns=['Customer_ID', 'Purchase']
    )

    y = df['Purchase']

    groups = df['Customer_ID']

    return X, y, groups


def encode_data(X):

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X


def split_train_test(X, y, groups):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    return X_train, X_test, y_train, y_test