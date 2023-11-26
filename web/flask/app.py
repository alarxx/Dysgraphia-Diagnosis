from flask import Flask, request, jsonify, render_template
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn

app = Flask(__name__)

# Определение модифицированной модели
def create_model(num_classes):
    model = models.vgg16(pretrained=False)  # pretrained=False, так как мы загружаем свои веса
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

# Замените это на количество классов в вашем датасете
num_classes = 2  # Пример
vgg16 = create_model(num_classes)

# Загрузка весов модели
vgg16.load_state_dict(torch.load('model_fold1.pth'))
vgg16.eval()

# Функция для обработки изображения
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))

    # Преобразование в RGB, если изображение не в RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    # Отображение HTML-страницы
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'})
    file = request.files['file']
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    outputs = vgg16(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return jsonify({'prediction': predicted_idx})

if __name__ == '__main__':
    app.run(debug=True)
