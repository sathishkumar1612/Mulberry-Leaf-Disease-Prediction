# 🔬 MORUS-LENS — Integration & Deployment Guide

## System Components
| File / Directory | Purpose |
|------------------|---------|
| `app.py` | Production Flask backend with Authentication, History, Contact, and Botanical Vision Engine |
| `templates/` | Clean responsive templates (`dashboard.html`, `login.html`, `register.html`, `history.html`, `guide.html`, `contact.html`, `base.html`) |
| `static/css/style.css` | Light agronomic design system with mobile navigation drawer |
| `train_advanced_model.py` | EfficientNetB0 two-phase transfer learning pipeline |
| `instance/mulberry_app.db` | Auto-created SQLite database for users, predictions, and contact inquiries |
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

## STEP 2 — Start the Server
```bash
python app.py
```
You should see:
```text
[OK] Loaded Keras Model / Vision Engine
 * Running on http://127.0.0.1:5000
```

---

## STEP 3 — Access the Web Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- **Register / Sign In**: Register your private account to manage diagnoses and submit field notes.
- **Analyze**: Upload a leaf image on the dashboard or snap directly via mobile camera for immediate diagnostic results.
- **Field Guide**: Access the organic botanical guide (`/guide`) for symptom keys and silkworm safety timelines.
- **Contact**: Reach out to developer Sathish Kumar or submit inquiry tickets directly via `/contact`.

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
