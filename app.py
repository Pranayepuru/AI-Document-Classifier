import os
import json
import time
import csv
from io import StringIO
from flask import Flask, request, jsonify, render_template, send_file, make_response
from werkzeug.utils import secure_filename

from predict import DocumentPredictor
from train import train_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit
app.config['HISTORY_FILE'] = 'predictions_history.json'
app.config['STATS_FILE'] = 'model_stats.json'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
predictor = None
models_loaded = False

def init_predictor():
    global predictor, models_loaded
    try:
        predictor = DocumentPredictor()
        models_loaded = True
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        predictor = None
        models_loaded = False
        print(f"Warning: Models could not be loaded. Please train the model. Error: {e}")

# Initialize predictor on startup
init_predictor()

# Helper to read history
def get_history():
    if not os.path.exists(app.config['HISTORY_FILE']):
        return []
    try:
        with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

# Helper to save history
def save_history(history):
    try:
        with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

# Helper to get model stats
def get_model_stats():
    if not os.path.exists(app.config['STATS_FILE']):
        return {
            "total_documents": 0,
            "categories_count": 0,
            "best_model": "N/A",
            "accuracy": 0.0,
            "f1_score": 0.0,
            "models_comparison": {}
        }
    try:
        with open(app.config['STATS_FILE'], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global predictor, models_loaded
    
    if not models_loaded:
        # Try to initialize again in case models were trained since last request
        init_predictor()
        if not models_loaded:
            return jsonify({"error": "Model files are missing. Please retrain/train the model first."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Append unique timestamp to avoid name collisions
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Predict
            res = predictor.predict(filepath)
            
            # Record in history (including confidence for logging/debugging)
            history = get_history()
            history_item = {
                "id": timestamp_prefix,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                "filename": filename,
                "saved_filename": unique_filename,
                "prediction": res["prediction"],
                "confidence": res["confidence"],
                "processing_time": res["processing_time"],
                "preview": res["preview"]
            }
            # Insert at the beginning (recent first)
            history.insert(0, history_item)
            save_history(history)
            
            # Return prediction to UI without confidence
            return jsonify({
                "id": history_item["id"],
                "timestamp": history_item["timestamp"],
                "filename": history_item["filename"],
                "prediction": history_item["prediction"],
                "processing_time": history_item["processing_time"],
                "preview": history_item["preview"]
            })
            
        except Exception as e:
            # Cleanup failed upload
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Unsupported file format. Please upload .txt, .pdf, or .docx files."}), 400

@app.route('/train', methods=['POST'])
def train():
    global predictor, models_loaded
    try:
        train_model()
        init_predictor()
        stats = get_model_stats()
        return jsonify({
            "message": "Model training completed successfully.",
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def stats_endpoint():
    history = get_history()
    m_stats = get_model_stats()
    
    # Calculate category distribution in history
    dist = {}
    for item in history:
        cat = item["prediction"]
        dist[cat] = dist.get(cat, 0) + 1
        
    # Sanitize recent predictions to not include confidence
    sanitized_recent = []
    for item in history[:10]:
        sanitized_recent.append({
            "id": item.get("id"),
            "timestamp": item.get("timestamp"),
            "filename": item.get("filename"),
            "prediction": item.get("prediction"),
            "processing_time": item.get("processing_time"),
            "preview": item.get("preview")
        })

    return jsonify({
        "total_predictions": len(history),
        "model_stats": m_stats,
        "history_count": len(history),
        "distribution": dist,
        "recent_predictions": sanitized_recent
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global predictor, models_loaded
    if not models_loaded:
        init_predictor()
        if not models_loaded:
            return jsonify({"error": "Model files are missing. Please call /train endpoint."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Use parameter 'file'."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"api_temp_{int(time.time())}_{filename}")
        file.save(temp_filepath)
        
        try:
            res = predictor.predict(temp_filepath)
            # Cleanup temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            # Return API structured response without confidence
            return jsonify({
                "filename": res["filename"],
                "prediction": res["prediction"]
            })
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Unsupported file extension."}), 400



@app.route('/export_history', methods=['GET'])
def export_history():
    history = get_history()
    
    # Create an in-memory CSV
    si = StringIO()
    cw = csv.writer(si)
    
    # Header
    cw.writerow(['ID', 'Timestamp', 'Original Filename', 'Predicted Category', 'Confidence (%)', 'Processing Time (s)'])
    
    # Rows
    for item in history:
        cw.writerow([
            item.get('id', ''),
            item.get('timestamp', ''),
            item.get('filename', ''),
            item.get('prediction', ''),
            item.get('confidence', ''),
            item.get('processing_time', '')
        ])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=document_predictions_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    # Start flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
