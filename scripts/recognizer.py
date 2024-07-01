import torch
from PIL import Image
import torchvision.transforms as transforms

from main import Net

net = torch.load("GestRecogNet.pt")
net.eval()

img_path = "b3.jpg"

img = Image.open(img_path).convert('L')
img = img.resize((28, 28))

t = transforms.ToTensor()

img = t(img).unsqueeze(1).float()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img = img.to(device)

print(img.shape)

ans = net.forward(img)

print(ans.argmax(dim=1))