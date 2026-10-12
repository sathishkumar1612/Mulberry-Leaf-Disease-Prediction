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

## Production Deployment Options

### Option A — Deploy with Docker (Recommended)
Docker ensures the exact same dependencies (Python 3.11, OpenCV libraries, and Keras) run without compatibility bugs anywhere in the cloud or local servers.

```bash
# 1. Build the Docker container
docker build -t morus-lens .

# 2. Run the container with persistent volumes
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/instance:/app/instance \
  -v $(pwd)/static/uploads:/app/static/uploads \
  --name morus_lens_app \
  morus-lens
```

#### Or with Docker Compose (One Single Command):
```bash
docker compose up -d
```
- Your application will be live at: **`http://localhost:5000`**
- Farm databases (`instance/`) and uploaded leaf scans (`static/uploads/`) remain persistent even when the container restarts.

---

### Option B — Deploy to Free / Low-Cost Cloud (Render / Railway)

1. **Render.com**:
   - Push your code to GitHub.
   - In Render, click **New + Web Service** and link your `Mulberry-Leaf-Disease-Prediction` repository.
   - Choose **Docker** as the runtime (Render will automatically detect your `Dockerfile`).
   - Add a persistent disk mounted at `/app/instance` and `/app/static/uploads`.
   - Click **Deploy**! Render provides a free `https://morus-lens.onrender.com` SSL domain.

2. **Railway.app**:
   - Click **New Project** -> **Deploy from GitHub repo**.
   - Railway automatically detects the `Dockerfile` and builds the container.
   - In the settings, add a volume mounted at `/app/instance`.

---

### Option C — Traditional Linux VPS (Ubuntu + Gunicorn + Nginx)

1. **Install system dependencies**:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip nginx libgl1 libglib2.0-0
   ```
2. **Run with Gunicorn WSGI**:
   ```bash
   gunicorn --workers 3 --bind 127.0.0.1:5000 --timeout 120 app:app
   ```
3. **Configure Nginx as reverse proxy** to forward port 80/443 to `127.0.0.1:5000`.

---

*MORUS-LENS · Built with Flask + SQLite + OpenCV + EfficientNetB0 · Engineered by Sathish Kumar*
