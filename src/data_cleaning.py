import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    return df


def clean_data(df):

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Fill missing numerical values with median
    numerical_columns = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    for column in numerical_columns:

        if column != 'Purchase':
            df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values with mode
    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    for column in categorical_columns:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    return df