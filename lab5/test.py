import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Загрузка обученной модели
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

# Первый тестовый набор данных
def test_one():
  test_x_1 = np.linspace(0, 15, 100)
  test_y_1 = test_x_1 + np.random.random(100)*3-2
  test_x_1 = test_x_1.reshape(-1,1)
  test_y_1 = test_y_1.reshape(-1,1)
  assert model.score(test_x_1, test_y_1) > 0.5, "Низкий уровень предсказания!"

# Второй тестовый набор данных
def test_two():
  test_x_2 = np.linspace(0, 10, 100)
  test_y_2 = test_x_2 + np.random.random(100)*4-3
  test_x_2 = test_x_2.reshape(-1,1)
  test_y_2 = test_y_2.reshape(-1,1)
  assert model.score(test_x_2, test_y_2) > 0.5, "Низкий уровень предсказания!"

# Шумовой набор данных
def test_with_noise():
  noise_data_x = np.linspace(0, 15, 100)
  noise_data_y = noise_data_x + np.random.random(100)*3-1
  noise_data_y[25:45] *= -2
  noise_data_x = noise_data_x.reshape(-1,1)
  noise_data_y = noise_data_y.reshape(-1,1)
  assert model.score(noise_data_x, noise_data_y) > 0.5, "Низкий уровень предсказани"
