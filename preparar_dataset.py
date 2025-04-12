import os, shutil
from sklearn.model_selection import train_test_split

IMG_PATH = 'dataset/con_celular/images'
LBL_PATH = 'dataset/con_celular/labels'


train_img_path = 'dataset_yolo/images/train'
val_img_path = 'dataset_yolo/images/val'
train_lbl_path = 'dataset_yolo/labels/train'
val_lbl_path = 'dataset_yolo/labels/val'

os.makedirs(train_img_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(train_lbl_path, exist_ok=True)
os.makedirs(val_lbl_path, exist_ok=True)

images = [f for f in os.listdir(IMG_PATH) if f.endswith(('.jpg', '.png'))]
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

for img in train_imgs:
    label = img.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(IMG_PATH, img), os.path.join(train_img_path, img))
    shutil.copy(os.path.join(LBL_PATH, label), os.path.join(train_lbl_path, label))

for img in val_imgs:
    label = img.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(IMG_PATH, img), os.path.join(val_img_path, img))
    shutil.copy(os.path.join(LBL_PATH, label), os.path.join(val_lbl_path, label))

yaml_content = """path: dataset_yolo
train: images/train
val: images/val
names:
  0: persona
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ Dataset preparado y archivo 'data.yaml' generado.")
