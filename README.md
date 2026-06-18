# AI Document Classifier 🚀

A complete, professional, internship-level Machine Learning web application that automatically classifies uploaded documents (PDF, DOCX, TXT) into distinct categories with high confidence.

## 🌟 Features

- **Document Analysis**: Supports uploading and parsing of `.pdf`, `.docx`, and `.txt` files.
- **Machine Learning Models**: Trains and compares **Multinomial Naive Bayes** and **Logistic Regression**, automatically selecting the best-performing model based on F1-score/Accuracy.
- **Advanced Text Preprocessing**: Utilizes NLTK for lowercasing, tokenization, stopword removal, punctuation stripping, and lemmatization.
- **Feature Extraction**: Implements TF-IDF vectorization with bi-grams for semantic understanding.
- **Modern Dashboard UI**:
  - Premium dark/light mode toggle with persistent local storage settings.
  - Interactive cards showing processed documents count, active model metrics, supported categories, and training dataset size.
  - Real-time Chart.js analytics: Category Distribution (Doughnut chart) and Model Metrics Comparison (Bar chart).
  - Elegant drag-and-drop file upload zone with animated progress indicators.
- **Prediction History**: 
  - Real-time client-side search to filter previous classifications.
  - Downloadable professional PDF reports containing analysis results and text previews.
  - CSV export of entire prediction log history.
- **Dataset Auto-Generator**: Automatically creates a balanced synthetic corpus of 60 documents if the dataset folder is missing, allowing immediate run.
- **On-Demand Retraining**: Retrain the model on updated files directly from the UI with real-time dashboard statistic updates.

---

## 📂 Project Structure

```
document_classifier/
├── app.py                 # Flask main application
├── train.py               # Dataset generation and ML model training
├── predict.py             # Text extraction and model inference pipeline
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
├── model.pkl              # Saved best ML model (generated on train)
├── vectorizer.pkl         # Saved TF-IDF Vectorizer (generated on train)
├── model_stats.json       # Evaluated metrics of trained models
├── predictions_history.json # Persistent prediction history logs
├── dataset/               # Document training corpus folder
│   ├── Resume/
│   ├── Invoice/
│   ├── Report/
│   ├── Letter/
│   ├── Legal/
│   └── Research/
├── templates/
│   └── index.html         # Frontend Dashboard page
├── static/
│   ├── style.css          # Theme styles (gradients, dark mode, animations)
│   ├── script.js          # Interactive dashboard JavaScript logic
│   └── uploads/           # Uploaded files directory
└── screenshots/           # Application screenshots placeholder
```

---

## 🚀 Installation & Setup

### 1. Clone the Project
Navigate to your project directory.

### 2. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

*Note: If python is not on your global environment path, run using:*
```bash
python -m pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model
Run the training script to auto-generate the synthetic dataset (if missing), run text preprocessing, train Naive Bayes & Logistic Regression models, select the best model, and save files:
```bash
python train.py
```

### 4. Launch the Web Application
Start the Flask development server:
```bash
python app.py
```


---

## 📊 API Reference

The project exposes a JSON REST API for integration:

### Classify Document
- **Endpoint**: `/api/predict`
- **Method**: `POST`
- **Payload**: Form-data with key `file` containing the document file.
- **Response**:
```json
{
  "filename": "my_resume.pdf",
  "prediction": "Resume",
  "confidence": "96.4%"
}
```

---

## ☁️ Deployment Guide

### Deploying to Render
1. Create a Web Service on [Render](https://render.com/).
2. Connect your GitHub repository.
3. Select **Python** runtime environment.
4. Set Build Command:
   ```bash
   pip install -r requirements.txt && python train.py
   ```
5. Set Start Command:
   ```bash
   gunicorn app:app
   ```
6. Add environment variable `PORT = 10000` if needed.

### Deploying to Railway
1. Create a project on [Railway](https://railway.app/).
2. Connect your GitHub repository.
3. Railway will auto-detect Python. Add a `Procfile` if necessary:
   ```
   web: python train.py && gunicorn app:app
   ```
4. Set port to `0.0.0.0` environment variable injection.

---

## 🛠️ Future Enhancements
- Integrate Deep Learning (BERT or DistilBERT) for semantic classification.
- Support OCR using Tesseract for scanned PDF images.
- Multi-label classification for documents fitting multiple categories.
