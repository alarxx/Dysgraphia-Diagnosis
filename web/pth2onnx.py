import torch
import torchvision.models as models

# Загрузите предобученную модель VGG16
model = models.vgg16(pretrained=True)

# Убедитесь, что модель находится в режиме evaluation
model.eval()

# Создайте пример входных данных
dummy_input = torch.randn(1, 3, 224, 224)

# Экспортируйте модель в формат ONNX
torch.onnx.export(model, dummy_input, "model_vgg16.onnx", verbose=True)
