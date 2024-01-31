from catboost.datasets import titanic

if __name__ == '__main__':

    train_df, test_df = titanic()

    train_df = train_df[['Pclass', 'Sex', 'Age']]
    train_df.to_csv('titanic.csv', index=False)
