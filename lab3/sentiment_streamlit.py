from transformers import pipeline
import streamlit as st
from langid import classify


# создаем пайплайн для определения тональности текста
def get_pipeline():
    pipe = pipeline("sentiment-analysis")
    return pipe


def sentiment_and_score(inputs):
    """Выполняет определение тональности текста и возвращает метку и оценку.

    Args:
        inputs (str): Текст для определения тональности.

    Returns:
        tuple: Кортеж, в котором 1й элемент - метка тональности, а 2й - оценка.

    """
    pipe = get_pipeline()
    # в результате работы получаем лист,
    # в котором на 0 позиции стоит нужный нам dict
    result = pipe(inputs)[0]
    return result["label"], result["score"]


def language_test(inputs):
    """Проверяет, является ли язык текста английским.

    Args:
        inputs (str): Текст для проверки языка.

    Returns:
        bool: True, если язык английский, False в противном случае.

    """
    lang = classify(inputs)[0]
    return lang == "en"


# Определение тональности текста.
st.title("ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ ТЕКСТА.")
# Поле ввода текста. value - значение по умолчанию.
context = st.text_input("CONTEXT:", value="Life is awesome!")

# Кнопка, нажатие на которую запускает процесс определения тональности.
result = st.button("ОПРЕДЕЛИТЬ ТОНАЛЬНОСТЬ")
if context:
    if language_test(context):
        # распаковываем кортеж из тональности и оценки
        label, score = sentiment_and_score(context)
        st.text(f"LABEL={label}\nSCORE={score}")
    else:
        st.error("Язык не является английским")
else:
    st.error("Пустое поле ввода")
