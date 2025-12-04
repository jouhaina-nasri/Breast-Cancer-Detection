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

# 🌐 Web Interface (Screenshots)

### **Home Page**
![Home](https://github.com/user-attachments/assets/9464be53-a9c3-4180-82b6-a68281fa5357)

### **Upload Page**
![Upload](https://github.com/user-attachments/assets/076f4c1e-aab0-457d-9af0-36e560249252)

### **Prediction Result**
![Result](https://github.com/user-attachments/assets/710ccefb-ac58-4df1-b82c-3f78346ed13a)

### **Evaluation**
![Preview](https://github.com/user-attachments/assets/ab932640-3978-4fdd-8c43-81c7d7887f17)

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
