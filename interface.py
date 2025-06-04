import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os

# Класи (заміни, якщо інші)
class_names = ['lancet', 'supercam', 'zala']

# Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформації (повинні збігатись із тими, що були при валідації)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Функція передбачення
def predict_image(image_path, model_path):
    # Перевірка
    if not os.path.exists(image_path):
        print(f"[!] Файл {image_path} не знайдено.")
        return

    # Завантаження моделі
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Завантаження зображення
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Прогноз
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        print(f"✅ Передбачений клас: {predicted_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone Classifier Inference Script')
    parser.add_argument('image_path', type=str, help='Шлях до зображення')
    parser.add_argument('--model_path', type=str, default='model/best_model.pth', help='Шлях до моделі')

    args = parser.parse_args()
    predict_image(args.image_path, args.model_path)
