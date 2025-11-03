
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Customer Segmentation with K-Means")

import os
import joblib
import streamlit as st

# Абсолютный путь к файлам относительно этого скрипта
base_path = os.path.dirname(os.path.abspath(__file__))
kmeans_path = os.path.join(base_path, "kmeans_model.joblib")
scaler_path = os.path.join(base_path, "scaler.joblib")

# Проверка наличия файлов перед загрузкой
if not os.path.exists(kmeans_path):
    st.error(f"❌ Файл модели не найден по пути: {kmeans_path}")
else:
    kmeans = joblib.load(kmeans_path)

if not os.path.exists(scaler_path):
    st.error(f"❌ Файл scaler не найден по пути: {scaler_path}")
else:
    scaler = joblib.load(scaler_path)


# Загружаем данные
uploaded_file = st.file_uploader("Загрузите файл ccdata.csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Данные:")
    st.dataframe(data.head())

    # Убираем CUST_ID, если есть
    if 'CUST_ID' in data.columns:
        data = data.drop('CUST_ID', axis=1)

    # Заполняем пропуски медианой
    data = data.fillna(data.median())

    # Стандартизация с загруженным скейлером
    X_scaled = scaler.transform(data.select_dtypes(include=['float64', 'int64']))

    # Кластеризация с загруженной моделью
    data['Cluster'] = kmeans.predict(X_scaled)

    st.write("Результаты кластеризации:")
    st.dataframe(data.head())

    # Визуализация первых двух признаков
    fig, ax = plt.subplots()
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='tab10')
    ax.set_xlabel(data.select_dtypes(include=['float64', 'int64']).columns[0])
    ax.set_ylabel(data.select_dtypes(include=['float64', 'int64']).columns[1])
    ax.set_title("Кластеры клиентов")
    st.pyplot(fig)

