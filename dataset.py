import os
from PIL import Image
from torchvision import transforms


IMAGE_SIZE=224

def load_data(options):
    transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])
    ])
    
    with open(os.path.join(options.data_path, 'label.txt'), "r") as f:
        labels = f.readlines()
    for i in range(len(labels)):
        labels[i] = int(labels[i][29:-1])
    
    name = 'ILSVRC2012_val_000' + str(options.image_index+1).zfill(5) + '.JPEG'
    img_name = os.path.join(options.data_path,name)
    image=Image.open(img_name).convert('RGB')
    label=labels[options.image_index]
    return transform(image), label


