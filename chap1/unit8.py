import torch
import torch.onnx as onxx
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "data/modelWeight.pth")

model2 = models.vgg16()
model2.load_state_dict(torch.load("data/modelWeight.pth"))
model2.eval()

torch.save(model2, 'data/vgg_model.pth')
model3 = torch.load('data/vgg_model.pth')