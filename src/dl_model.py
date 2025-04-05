# Load and run DL model
import torch
from torchvision import models
from torchvision.transforms import v2
from PIL import Image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bestmodel.pth"
model = models.resnet50(weights=MODEL_PATH).to(device)
model.eval()

# Define transformation
class_means = [0.0163, 0.0159, 0.0145]
class_stds = [0.0025, 0.0025, 0.0026]

transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=class_means, std=class_stds),
        v2.Grayscale(num_output_channels=3)
    ])

# Predict with the model
def predict_image(cropped_cells):
    predictions = []
    for cell in cropped_cells:
        cell_tensor = Image.fromarray(cell)
        cell_tensor = transform(cell_tensor).unsqueeze(0)
    
        with torch.no_grad():
            output = model(cell_tensor.to(device))
            _, pred = torch.max(output, 1)
            predictions.append(pred.item())
    return predictions