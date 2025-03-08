import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import shutil
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

image_folder = "blood_cell_images/"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

IMG_SIZE = 64
image_data = []

for file in tqdm(image_files, desc="Loading Images"):
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten()
    image_data.append(img)

image_data = np.array(image_data)

NUM_CLUSTERS = 2
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(image_data)

labels = kmeans.labels_

output_folder = "clustered_images/"
for i in range(NUM_CLUSTERS):
    cluster_path = os.path.join(output_folder, f"class_{i}")
    os.makedirs(cluster_path, exist_ok=True)

for idx, file in enumerate(tqdm(image_files, desc="Organizing Images")):
    src_path = os.path.join(image_folder, file)
    cluster_path = os.path.join(output_folder, f"class_{labels[idx]}", file)
    shutil.copy(src_path, cluster_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(root=output_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Detected Classes:", dataset.classes)
