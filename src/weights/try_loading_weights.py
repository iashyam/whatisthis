import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import preprocess_image, labels

model = torch.load("burrah_mobilenet.pkl", weights_only=False)

print(model)
exit()
model.eval()
#torch.save(model, "app/burrah_mobilenet.pkl")

image_wp = Image.open('sample_images/hen.jpeg')
# image_wp = Image.open('sample_images/dog.jpg')
image = preprocess_image(image_wp).float()
output = torch.softmax(model(image), dim=1)
label = int(torch.argmax(output).item())
label = labels[label]
print(f"Predicted {label} with {torch.max(output)*100:1f}% probablity.") 
plt.imshow(image_wp)
plt.title(f"Predicted {label}")
plt.axis("off")
plt.show()
