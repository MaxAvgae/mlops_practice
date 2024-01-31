import pandas as pd


if __name__ == '__main__':

    train_df = pd.read_csv('titanic.csv')

    train_df = pd.get_dummies(train_df, columns=['Sex'])
    train_df.to_csv('titanic.csv', index=False)
