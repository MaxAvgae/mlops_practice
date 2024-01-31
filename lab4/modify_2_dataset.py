import pandas as pd

if __name__ == '__main__':
    # Считываем данные из файла
    train_df = pd.read_csv('titanic.csv')

    # Создаем новый столбец 'Sex' на основе 'Sex_female' и 'Sex_male'
    train_df['Sex'] = train_df['Sex_female'].map({True: 'female', False: 'male'})

    # Удаляем ненужные столбцы 'Sex_female' и 'Sex_male'
    train_df = train_df.drop(['Sex_female', 'Sex_male'], axis=1)

    # Сохраняем измененные данные
    train_df.to_csv('titanic.csv', index=False)
