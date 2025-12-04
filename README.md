# 🩺 Breast Cancer Detection — Deep Learning + Flask

This project uses a deep convolutional neural network (CNN) to classify breast tumor images into two categories:

**Malignant (Cancer)**  
**Normal**

It includes:

- A **complete training pipeline** (MobileNetV2 transfer learning)  
- A **Keras 3–compatible saved model** (`.keras`)  
- A **Flask web interface** to upload images and get predictions  
- A **global evaluation route** (`/evaluate`) returning accuracy + confusion matrix  

---

## 📁 Project Structure

Breast-Cancer-Detection/
│
├── app/
│   ├── app.py                 # Flask backend
│   ├── config.py              # Project settings
│   ├── inference.py           # Prediction + evaluation logic
│
├── templates/
│   ├── index.html             # Home page
│   └── test.html              # Upload + results page
│
├── static/
│   └── assets/                # CSS / JS / images
│
├── data/
│   ├── training_set/          # Training dataset
│   │   ├── maligne/
│   │   └── normal/
│   └── test_set/              # Evaluation dataset
│       ├── maligne/
│       └── normal/
│
├── models/
│   └── breast_cancer_cnn.keras  # Saved model (Keras 3)
│
├── training/
│   └── train.py               # Training script
│
└── requirements.txt

---

# 🌐 Web Interface (Screenshots)

### **Home Page**
![Home](https://user-images.githubusercontent.com/63677147/206879064-385dd5e4-087b-4fc4-a6ed-2635802c7c0c.jpg)

### **Upload Page**
![Upload](https://user-images.githubusercontent.com/63677147/206879075-faacd142-a8a1-4a64-b457-0f370ec81193.jpg)

### **Preview Images**
![Preview](https://user-images.githubusercontent.com/63677147/206879089-d86303d0-b4fb-4375-8a48-42f77df0b6fd.jpg)

### **Prediction Result**
[![Result](https://user-images.githubusercontent.com/63677147/206879108-4fe350ae-aad5-4061-b297-80f87f5dd77e.jpg)](https://github.com/user-attachments/assets/710ccefb-ac58-4df1-b82c-3f78346ed13a" />)

---

# ⚙️ Installation & Running

### **1️⃣ Create a virtual environment**
```bash
python -m venv venv
```
### **2️⃣ Activate it**
***Windows***
```bash
venv\Scripts\activate
```
***Linux / macOS***
```bash
source venv/bin/activate
```
### **3️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```
### **4️⃣ Train the model (optional)**
```bash
python training/train.py
```
### **5️⃣ Run the Flask application**
```bash
python app/app.py
```
Then open your browser:
➡️ http://127.0.0.1:5000

---

### **🤖 Model Training**

The model is a fine-tuned MobileNetV2 network:

- Input: 224 × 224 × 3 images
- Preprocessing: MobileNetV2 preprocess_input
- Loss: Binary Crossentropy
- Optimizer: Adam (1e-4)
- Output: sigmoid (probability of Normal)

💾 Model saved as:
```bash
models/breast_cancer_cnn.keras
```
