# 🍃 Mulberry Leaf Disease Detector — Integration & Deployment Guide

## System Components
| File / Directory | Purpose |
|------------------|---------|
| `app.py` | Production Flask backend with Authentication, History, and Inference |
| `templates/` | Clean light professional dashboard templates (`dashboard.html`, `login.html`, `register.html`, `history.html`, `guide.html`, `base.html`) |
| `static/css/style.css` | Light medical/agricultural design system |
| `train_advanced_model.py` | EfficientNetB0 two-phase transfer learning pipeline |
| `instance/mulberry_app.db` | Auto-created SQLite database for users and predictions |
| `requirements.txt` | Production dependencies |
| `mulberry_leaf_disease_Identification.keras` | Pre-trained model file |

---

## Step-by-Step Setup

### STEP 1 — Virtual Environment & Dependencies
```bash
python -m venv myenv
# Activate on Windows:
myenv\Scripts\activate

# Install requirements:
pip install -r requirements.txt
```

---

### STEP 2 — Start the Server
```bash
python app.py
```
You should see:
```text
[OK] Loaded Keras Model: mulberry_leaf_disease_Identification.keras (Input shape: (128, 128))
 * Running on http://127.0.0.1:5000
```

---

### STEP 3 — Access the Web Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- **Login**: Use the seeded demo credentials (`demo` / `demo1234`) or register a new account.
- **Analyze**: Upload a leaf image on the dashboard to see immediate diagnostic results and sericulture protocols.
- **History**: Review past scans and track pathological developments over time.

---

## Retraining With EfficientNetB0
To train on custom datasets with heavy data augmentation and balanced class weights:
```bash
python train_advanced_model.py \
  --train_dir path/to/dataset/train \
  --val_dir path/to/dataset/val \
  --output mulberry_leaf_disease_Identification_v2.keras
```

---

## Production Deployment Checklist
1. **WSGI Server**: Use `gunicorn` or `waitress`:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
2. **Secret Key**: Set `SECRET_KEY` environment variable in production.
3. **Database**: SQLite is configured by default; set `DATABASE_URL` if connecting to PostgreSQL / MySQL.

---

*Built with Flask + Flask-Login + SQLite + TensorFlow/Keras · EfficientNetB0 & CNN Transfer Learning*
