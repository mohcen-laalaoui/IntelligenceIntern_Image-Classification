# IntelligenceIntern_Image-Classification
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
**IntelligenceIntern_Image-Classification** is a deep learning project that builds an **image classification model** from scratch using **PyTorch**. The project includes **data preprocessing, model development, training, and evaluation** to classify images into two categories. This project is designed to handle unlabeled image datasets by applying clustering techniques to create class labels before training a Convolutional Neural Network (CNN).

## 🚀 Features
- 📂 **Data Preprocessing**: Loads and preprocesses images with automatic clustering for unlabeled datasets.  
- 🧠 **CNN Model Development**: Implements a deep learning model using **PyTorch** for classification.  
- 🎯 **Training & Optimization**: Uses **CrossEntropyLoss** and **Adam optimizer** to improve accuracy.  
- 📊 **Evaluation**: Measures model performance using accuracy and loss metrics.  
- ⚡ **Deployment Ready**: The trained model can be integrated into real-world applications.

## 📂 Project Structure
```
IntelligenceIntern_Image-Classification/
👉 data/                     # Contains image dataset
👉 Deep Learning Model Development.ipynb  # Jupyter Notebook with full code
👉 README.md                 # Project documentation
```

## ⚙️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohcen-laalaoui/IntelligenceIntern_Image-Classification.git
   cd IntelligenceIntern_Image-Classification
   ```
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏠 Model Development  
This project uses a **Convolutional Neural Network (CNN)** for image classification.  
If the dataset is **unlabeled**, K-Means clustering is applied to automatically generate two classes before training.

### 🔹 Data Preprocessing
- Loads images from the dataset folder.  
- Applies image transformations such as resizing, normalization, and augmentation.  
- If labels are missing, **K-Means clustering** is used to separate images into two categories.  

### 🔹 CNN Architecture
The Convolutional Neural Network (CNN) is designed with multiple convolutional layers, ReLU activation, max pooling, and fully connected layers.  
```python
class BloodCellCNN(nn.Module):
    def __init__(self):
        super(BloodCellCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.dropout = nn.Dropout(0.6)  # Updated Dropout from 0.5 to 0.6
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 🔹 Training & Optimization
- **Loss Function**: The model uses `CrossEntropyLoss` since this is a classification problem.  
- **Optimizer**: `Adam` optimizer is used for efficient parameter tuning.  
- **Training Loop**:
  - The dataset is divided into training and validation sets.  
  - Images are passed through the CNN, and gradients are updated based on the loss.  
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 🔹 Model Evaluation
After training, the model is evaluated on the test dataset using accuracy, precision, recall, and loss metrics.
```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
```

## 📊 Results
- The model achieves **high accuracy** after training on labeled or automatically clustered images.
- Accuracy improves with **data augmentation and hyperparameter tuning**.
- The trained model can be used for **blood cell classification or other medical image tasks**.

## 📌 Future Improvements
- 🔥 **Fine-tune with a pre-trained model** (e.g., ResNet, VGG for better performance).  
- 🖼️ **Expand dataset** to handle more image classes.  
- 🚀 **Deploy model** using Flask, FastAPI, or streamlit for real-world applications.  

## 🤝 Contributing
Feel free to contribute by improving the model, adding new features, or optimizing performance.

**Steps to contribute:**
1. Fork the repository.  
2. Create a new branch (`git checkout -b new-feature`).  
3. Commit your changes (`git commit -m "Add new feature"`).  
4. Push to your fork and submit a pull request.  

---
**Recent Updates:**
- Increased dropout from `0.5` to `0.6` for better regularization.
- Enhanced DataLoader setup for more efficient training.
- Improved training performance with refined hyperparameters.

## 🔗 **References**
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- K-Means Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Deep Learning with PyTorch: https://pytorch.org/tutorials/
