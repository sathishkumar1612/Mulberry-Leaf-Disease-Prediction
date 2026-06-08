# 🍃 Mulberry Leaf Disease Detector — Integration Guide

## What You Have
| File | Purpose |
|------|---------|
| `app.py` | Flask backend — loads your model and handles predictions |
| `templates/index.html` | Beautiful web UI |
| `requirements.txt` | Python packages to install |

---

## Step-by-Step Setup (No Experience Needed)

### STEP 1 — Organize Your Project Folder

Create a folder called `mulberry_app` on your computer. Inside it, place these files:

```
mulberry_app/
├── app.py
├── requirements.txt
├── mulberry_leaf_disease_Identification.keras   ← YOUR MODEL FILE
└── templates/
    └── index.html
```

> ⚠️ The `.keras` model file must be **directly inside** `mulberry_app/` — same level as `app.py`

---

### STEP 2 — Install Python Packages

Open **Anaconda Prompt** (or any terminal) and run:

```bash
pip install flask tensorflow pillow numpy
```

Wait for it to finish (may take 2–3 minutes).

---

### STEP 3 — Run the App

In your terminal, navigate to your project folder:

```bash
cd path\to\mulberry_app
```

Example (adjust to your path):
```bash
cd C:\Users\SATHISH\Desktop\mulberry_app
```

Then start the server:

```bash
python app.py
```

You should see output like:
```
✅ Model loaded successfully!
 * Running on http://127.0.0.1:5000
```

---

### STEP 4 — Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000
```

You'll see the leaf disease detector web page! 🎉

---

### STEP 5 — How to Use

1. Click **"Choose Image"** or drag & drop a mulberry leaf photo
2. Click **"🔬 Analyze Leaf"**
3. See the **diagnosis**, **confidence %**, and **recommendations**

---

## What the App Does

| Feature | Detail |
|---------|--------|
| Image input size | Resized to 128×128 (matches your training) |
| Normalization | Divided by 255.0 (matches your rescale=1./255) |
| Classes | Disease Free · Potential Leaf Rust · Potential Leaf Spot |
| Output | Predicted class + confidence % for all 3 classes |
| Recommendations | Specific treatment advice per disease |

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `Model not found` error | Make sure `.keras` file is in the same folder as `app.py` |
| `pip not found` | Use Anaconda Prompt, not regular CMD |
| Port already in use | Change `port=5000` to `port=5001` in `app.py` |
| TensorFlow version mismatch | Run `pip install tensorflow==2.20.0` |
| Page not loading | Make sure `python app.py` is still running in the terminal |

---

## To Stop the Server

Press `Ctrl + C` in the terminal.

---

## Optional: Share With Others on Your Network

Change the last line of `app.py` to:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

Then others on the same WiFi can access it at `http://<your-ip>:5000`

---

*Built with Flask + TensorFlow/Keras · CNN Model · 3-Class Leaf Disease Classification*
