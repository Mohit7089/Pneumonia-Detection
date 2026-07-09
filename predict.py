import torch
from PIL import Image
from torchvision import transforms
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model()

model.load_state_dict(
    torch.load("models/pneumonia_pt_best.pth", map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        probability = model(image).item()

    pneumonia = probability
    normal = 1 - probability

    return {
        "NORMAL": normal,
        "PNEUMONIA": pneumonia
    }