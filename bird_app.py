
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import requests
import time
import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from io import BytesIO
from PIL import Image
from torchvision.models import resnet18


# ==========================================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ==========================================

st.set_page_config(
    page_title="Классификация птиц",
    page_icon="🐦",
    layout="wide"
)


# Пути к файлам
project_dir = Path(__file__).resolve().parent

model_path = project_dir / "models" / "bird_resnet18.pth"
metrics_path = project_dir / "models" / "bird_metrics.json"

device = torch.device("cpu")


# ==========================================
# БОКОВОЕ МЕНЮ
# ==========================================

st.sidebar.title("🐦 Птицы")

page = st.sidebar.radio(
    "Выберите раздел:",
    ["Классификация птиц", "Метрики модели"]
)


# ==========================================
# ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЙ
# ==========================================

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# ЗАГРУЗКА МОДЕЛИ
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            in_features=512,
            out_features=len(class_names)
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names


# ==========================================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ==========================================

def predict_image(image, model, class_names):

    image = image.convert("RGB")

    start_time = time.time()

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = probabilities.max(dim=1)

    prediction_time = time.time() - start_time

    predicted_class = class_names[predicted.item()]

    confidence = confidence.item()

    return predicted_class, confidence, prediction_time


# ==========================================
# СТРАНИЦА 1. КЛАССИФИКАЦИЯ ПТИЦ
# ==========================================

if page == "Классификация птиц":

    st.title("🐦 Определение вида птицы")

    st.write(
        "Загрузите фотографию птицы или укажите ссылку на изображение. "
        "Модель ResNet18 попробует определить её вид."
    )

    # Проверяем наличие модели
    if not model_path.exists():

        st.error(f"Файл модели не найден: {model_path}")

        st.stop()

    # Загружаем модель
    model, class_names = load_model()

    # Загрузка фотографий
    uploaded_files = st.file_uploader(
        "Загрузите фотографии птиц",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # Загрузка по ссылке
    image_url = st.text_input(
        "Или вставьте ссылку на изображение:"
    )

    images = []

    # Добавляем загруженные фотографии
    if uploaded_files:

        for uploaded_file in uploaded_files:

            image = Image.open(uploaded_file)

            images.append(
                (uploaded_file.name, image)
            )

    # Загружаем изображение по URL
    if image_url:

        try:

            response = requests.get(
                image_url,
                timeout=15
            )

            response.raise_for_status()

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            images.append(
                ("Изображение по ссылке", image)
            )

        except Exception as error:

            st.error(
                f"Не удалось загрузить изображение: {error}"
            )

    # Классификация изображений
    if images:

        st.subheader("Результаты классификации")

        for name, image in images:

            st.divider()

            st.write(f"**{name}**")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption=name,
                    use_container_width=True
                )

            with col2:

                predicted_class, confidence, prediction_time = predict_image(
                    image,
                    model,
                    class_names
                )

                st.success(
                    f"Предсказанный вид: {predicted_class}"
                )

                st.metric(
                    "Уверенность модели",
                    f"{confidence:.2%}"
                )

                st.metric(
                    "Время обработки",
                    f"{prediction_time:.3f} сек"
                )


# ==========================================
# СТРАНИЦА 2. МЕТРИКИ МОДЕЛИ
# ==========================================

elif page == "Метрики модели":

    st.title("📊 Результаты обучения модели птиц")

    # Проверяем наличие файла с метриками
    if not metrics_path.exists():

        st.error(f"Файл метрик не найден: {metrics_path}")

        st.stop()

    # Загружаем метрики
    with open(metrics_path, "r", encoding="utf-8") as file:

        metrics = json.load(file)

    # Проверяем наличие необходимых данных
    required_keys = [
        "training_time",
        "dataset_size",
        "train_size",
        "valid_size",
        "f1_score",
        "train_losses",
        "valid_losses",
        "train_accs",
        "valid_accs",
        "class_distribution",
        "class_names",
        "confusion_matrix"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in metrics
    ]

    if missing_keys:

        st.error(
            "В файле bird_metrics.json отсутствуют данные: "
            + ", ".join(missing_keys)
        )

        st.stop()

    # ======================================
    # ОСНОВНЫЕ ПОКАЗАТЕЛИ
    # ======================================

    st.subheader("Основные показатели")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Время обучения",
            f'{metrics["training_time"] / 60:.1f} мин'
        )

    with col2:

        st.metric(
            "Размер датасета",
            metrics["dataset_size"]
        )

    with col3:

        st.metric(
            "F1-score",
            f'{metrics["f1_score"]:.3f}'
        )

    st.write(
        f'Обучающая выборка: {metrics["train_size"]} изображений'
    )

    st.write(
        f'Проверочная выборка: {metrics["valid_size"]} изображений'
    )

    st.divider()

    # ======================================
    # ГРАФИКИ LOSS И ACCURACY
    # ======================================

    st.subheader("Графики обучения")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Loss**")

        fig, ax = plt.subplots(figsize=(6, 4))

        epochs = range(
            1,
            len(metrics["train_losses"]) + 1
        )

        ax.plot(
            epochs,
            metrics["train_losses"],
            marker="o",
            label="Train"
        )

        ax.plot(
            epochs,
            metrics["valid_losses"],
            marker="o",
            label="Validation"
        )

        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        fig.tight_layout()

        # st.pyplot(
        #     fig,
        #     use_container_width=True
        # )

        st.pyplot(fig, width="stretch")

        plt.close(fig)

    with col2:

        st.write("**Accuracy**")

        fig, ax = plt.subplots(figsize=(6, 4))

        epochs = range(
            1,
            len(metrics["train_accs"]) + 1
        )

        ax.plot(
            epochs,
            metrics["train_accs"],
            marker="o",
            label="Train"
        )

        ax.plot(
            epochs,
            metrics["valid_accs"],
            marker="o",
            label="Validation"
        )

        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid()
        fig.tight_layout()

        # st.pyplot(
        #     fig,
        #     use_container_width=True
        # )
        st.pyplot(fig, width="stretch")

        plt.close(fig)

    st.divider()

    # ======================================
    # РАСПРЕДЕЛЕНИЕ ИЗОБРАЖЕНИЙ ПО КЛАССАМ
    # ======================================
    # Распределение изображений по классам
    # Распределение изображений по классам
    st.subheader("Распределение изображений по видам птиц")

    class_distribution = metrics["class_distribution"]

    # Сортируем классы по количеству изображений
    sorted_classes = sorted(
        class_distribution.items(),
        key=lambda x: x[1]
    )

    fig, ax = plt.subplots(figsize=(10, 20))

    ax.barh(
        [class_name for class_name, count in sorted_classes],
        [count for class_name, count in sorted_classes]
    )

    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.margins(y=0)
    
    ax.set_xlim(left=0)

    ax.set_xlabel("Количество изображений")
    ax.set_ylabel("Виды птиц")
    ax.set_title("Распределение изображений по классам")

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # ======================================
    # МАТРИЦА ОШИБОК
    # ======================================

    import plotly.graph_objects as go
    from streamlit_plotly_events import plotly_events

    st.subheader("Матрица ошибок")

    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            hovertemplate=
                "Истинный класс: %{y}<br>" +
                "Предсказанный класс: %{x}<br>" +
                "Количество: %{z}<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Предсказанный класс",
        yaxis_title="Истинный класс",
        height=900,
        # title="Матрица ошибок"
    )

    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False
    },
    key="bird_confusion_matrix"
    )

    # st.plotly_chart(
    #     fig,
    #     use_container_width=True,
    #     config={"scrollZoom": True}
    # )

    st.divider()