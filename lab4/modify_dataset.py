import pandas as pd


if __name__ == '__main__':

    train_df = pd.read_csv('titanic.csv')

    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
    train_df.to_csv('titanic.csv', index=False)
