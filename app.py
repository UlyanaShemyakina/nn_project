
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import requests
import time
import json
import matplotlib.pyplot as plt

from pathlib import Path
from io import BytesIO
from PIL import Image
from torchvision.models import resnet18



# Настройки приложения
st.set_page_config(
    page_title="Weather Classification",
    page_icon="🌦️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "weather_resnet18.pth"
METRICS_PATH = PROJECT_DIR / "models" / "weather_metrics.json"

device = torch.device("cpu")

# Преобразование изображения
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Загрузка модели
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


# Предсказание
def predict_image(image, model, class_names):
    image_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    elapsed_time = time.perf_counter() - start_time

    predicted = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted].item()

    return class_names[predicted], confidence, elapsed_time



# Страница классификации
def classification_page():
    st.title("🌦️ Классификация погодных явлений")

    st.write(
        "Загрузите фотографию или вставьте ссылку, "
        "чтобы определить погодное явление."
    )

    model, class_names = load_model()

    st.subheader("Загрузка изображений")

    uploaded_files = st.file_uploader(
        "Выберите одну или несколько фотографий",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    image_url = st.text_input(
        "Или вставьте ссылку на изображение"
    )

    if st.button("Определить погоду"):
        images = []

        # Загруженные фотографии
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                images.append((uploaded_file.name, image))
            except Exception as error:
                st.error(
                    f"Не удалось открыть {uploaded_file.name}: {error}"
                )

        # Фотография по ссылке
        if image_url.strip():
            try:
                response = requests.get(
                    image_url.strip(),
                    timeout=15
                )
                response.raise_for_status()

                image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                images.append(("Изображение по ссылке", image))

            except Exception as error:
                st.error(f"Не удалось загрузить изображение по ссылке: {error}")

        if not images:
            st.warning("Сначала загрузите фотографию или вставьте ссылку.")

        # Классификация всех изображений
        for name, image in images:
            st.divider()
            st.subheader(name)

            st.image(image, use_container_width=True)

            predicted_class, confidence, elapsed_time = predict_image(
                image,
                model,
                class_names
            )

            st.success(f"Предсказанный класс: {predicted_class}")
            st.write(f"Уверенность модели: {confidence:.2%}")
            st.write(f"Время обработки: {elapsed_time:.3f} сек.")

# Страница информации о модели
def model_info_page():
    st.title("📊 Информация о модели")

    if not METRICS_PATH.exists():
        st.warning(
            "Файл с метриками пока не найден. "
            "Сначала сохрани результаты обучения из ноутбука."
        )
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.subheader("Процесс обучения")

    # Графики Loss и Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(metrics["train_losses"], label="Train Loss")
    axes[0].plot(metrics["valid_losses"], label="Valid Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(metrics["train_accs"], label="Train Accuracy")
    axes[1].plot(metrics["valid_accs"], label="Valid Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    # Время обучения и размер датасета
    st.subheader("Характеристики датасета")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Всего изображений",
        metrics["dataset_size"]
    )

    col2.metric(
        "Обучающих изображений",
        metrics["train_size"]
    )

    col3.metric(
        "Валидационных изображений",
        metrics["valid_size"]
    )

    if metrics["training_time"] is not None:
        st.metric(
            "Время обучения",
            f'{metrics["training_time"]:.1f} сек.'
        )
    else:
        st.info("Время обучения не было сохранено.")

    # Распределение классов
    st.subheader("Распределение изображений по классам")

    class_names = list(metrics["class_distribution"].keys())
    class_counts = list(metrics["class_distribution"].values())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(class_names, class_counts)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество изображений")
    ax.tick_params(axis="x", rotation=45)

    st.pyplot(fig)
    plt.close(fig)

    # F1-score
    # st.subheader("Качество модели")

    # st.metric(
    #     "F1-score - взвешенная качество классификации (баланс между точностью и полнотой)",
    #     f'{metrics["f1_score"]:.3f}'
    # )

    st.subheader("Качество модели")

    st.metric(
    "F1-score (взвешенное среднее)",
    f'{metrics["f1_score"]:.3f}'
)

    st.latex(
    r"F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}"
    )

    st.write(
    "F1-score — гармоническое среднее Precision и Recall. "
    "Показывает баланс между точностью и полнотой классификации."
    )


    # Confusion Matrix
    # Матрица ошибок
    st.subheader("Матрица ошибок")

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        metrics["confusion_matrix"],
        cmap="Blues"
    )

    ax.set_xticks(range(len(metrics["class_names"])))
    ax.set_yticks(range(len(metrics["class_names"])))

    ax.set_xticklabels(
        metrics["class_names"],
        rotation=45,
        ha="right",
        fontsize=7
    )

    ax.set_yticklabels(
        metrics["class_names"],
        fontsize=7
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Добавляем цифры в ячейки
    for i in range(len(metrics["class_names"])):
        for j in range(len(metrics["class_names"])):
            ax.text(
                j,
                i,
                metrics["confusion_matrix"][i][j],
                ha="center",
                va="center",
                fontsize=6
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    # st.pyplot(fig, use_container_width=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# Навигация между страницами
page = st.sidebar.radio(
    "Навигация",
    [
        "Классификация погоды",
        "Информация о модели"
    ]
)

if page == "Классификация погоды":
    classification_page()
else:
    model_info_page()
