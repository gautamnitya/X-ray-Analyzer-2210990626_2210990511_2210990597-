# 🏥 AI-Driven X-Ray Analysis System

Welcome to the AI-driven X-Ray Analysis System! This project walks you through building a complete, production-ready Machine Learning pipeline from scratch. It perfectly aligns with the techniques mentioned in our abstract: **Filtering, Contrast Adjustment (CLAHE), Edge Enhancement, and Normalization**, followed by a **Convolutional Neural Network (CNN)** for classification.

| 👥 Team Details | 📄 Project Information |
| :--- | :--- |
| • **Nitya Gautam** (2210990626)<br>• **Nandini Bakshi** (2210990597)<br>• **Khushi Bansal** (2210990511) | We have done a research paper on this and the result is awaited. |

## 🚀 Features
1. **End-to-end ML Pipeline**: Includes dataset loading, image enhancement, training, evaluation, and inference.
2. **Transfer Learning**: Uses an industry-standard ResNet18 model framework for fast, >95% accurate predictions.
3. **Explainable AI (XAI)**: Uses Grad-CAM to generate heatmaps showing exactly *where* the model is looking when it makes a diagnosis.
4. **Web Interface**: A clean, easy-to-use web app built with FastAPI.

---

## 🛠️ Step 1: Installation & Setup

1. **Install Python**: Make sure you have Python installed (preferably 3.9 or higher).
2. **Open Terminal**: Navigate to this directory (`c:\Users\nitya\Desktop\xray proj`).
3. **Install Dependencies**: Run the following command to gently install all required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Step 2: Download the Dataset

We need a dataset to train our Artificial Intelligence.
1. Go to **Kaggle** and download a standard Chest X-Ray dataset, such as:
   [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Create a folder named `data` inside your project directory.
3. Inside `data`, create two folders: `train` and `test`.
4. Inside `train` and `test`, place the images into two category folders: `NORMAL` and `ABNORMAL` (or rename the PNEUMONIA folder to ABNORMAL).

Your folder structure should look exactly like this:
```text
xray proj/
│── data/
│   ├── train/
│   │   ├── NORMAL/    <-- Place normal X-rays here
│   │   ├── ABNORMAL/  <-- Place pneumonia/abnormal X-rays here
│   ├── test/
│       ├── NORMAL/
│       ├── ABNORMAL/
```

## 🧠 Step 3: Train the Model

Once your data is in place, you can train the AI!
Run the training script from your terminal:
```bash
python -m src.train
```
- This will loop through the dataset, learning the difference between normal and abnormal X-rays.
- At the end, it creates `best_model.pth` (saving the 'brain' of the AI) and `training_curves.png` to visualize its learning progress.

> **How to improve accuracy if it doesn't reach 95%?**
> 1. Increase the `num_epochs` in `src/train.py` (e.g., from 10 to 20 or 30).
> 2. Add more data to your `train` folders.
> 3. Adjust the `learning_rate` inside `src/train.py`.

## 📊 Step 4: Evaluate the Model

To see how well your model performs on new data (Accuracy, Precision, Recall, F1-Score):
```bash
python -m src.evaluate
```
This script generates a `confusion_matrix.png`, showing exactly how many images it predicted right vs wrong.

## 🌐 Step 5: Run the Web App (UI)

Now for the fun part! You can upload custom X-rays to your very own website and get an instant diagnosis and explainability mask.
```bash
uvicorn app:app --reload
```
- Open your browser and go to: **[http://localhost:8000](http://localhost:8000)**
- Upload an image and see the results!

---
*Created by Nitya Gautam, Nandini Bakshi, and Khushi Bansal.*
