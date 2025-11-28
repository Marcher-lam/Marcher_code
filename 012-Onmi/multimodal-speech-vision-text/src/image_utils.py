from PIL import Image
import torchvision.transforms as T

img_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])



def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img)
    return img