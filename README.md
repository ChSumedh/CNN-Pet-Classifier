# 🐾 CNN Pet Classifier

A full-stack web app that identifies a pet's breed from a photo — 37 breeds total (25 dog breeds + 12 cat breeds) — trained on the Oxford-IIIT Pet Dataset. Beyond just serving predictions, it includes a **feedback loop**: users can flag wrong predictions, admins can review corrections, and the model can be **fine-tuned live on that corrected data** — right from the admin dashboard.

## ✨ What it does

- **Predicts breed from an uploaded image** using a transfer-learned EfficientNetB0 CNN, served via TensorFlow Lite for fast inference
- **User accounts** with hashed passwords and session-based auth
- **Feedback collection** — every prediction is logged; users can flag if it was wrong and submit the correct class
- **Admin dashboard** to review flagged predictions, promote users to admin, and trigger fine-tuning
- **Live model fine-tuning** — admins can retrain the model on accumulated corrected feedback directly from the browser, and the running app hot-swaps in the newly trained model without a redeploy

## 🧠 Model Architecture

Built on **EfficientNetB0** (pretrained on ImageNet) as a frozen feature extractor, with a small trainable classification head on top:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ efficientnetb0 (Functional)     │ (None, 7, 7, 1280)     │     4,049,571 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       163,968 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 37)             │         4,773 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 4,218,312 (16.09 MB)
 Trainable params: 168,741 (659.14 KB)
 Non-trainable params: 4,049,571 (15.45 MB)
```

**Why this design:**
- **Transfer learning** — EfficientNetB0's convolutional base stays frozen, so the model leverages ImageNet-scale feature representations without needing to train millions of parameters from scratch on a relatively small pet dataset.
- **Only ~169K trainable params** — keeps training fast and lightweight, which matters a lot for the live in-app fine-tuning feature, since it can run on a handful of user-submitted corrections without needing a GPU cluster.
- **Deployed as TFLite** — the trained Keras model is converted to `.tflite` for inference, which is smaller and faster to load in a live web request than the full Keras model.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| ML / Training | TensorFlow / Keras |
| Inference (serving) | TensorFlow Lite (`ai-edge-litert`) |
| Database | SQLAlchemy ORM → PostgreSQL (production) / SQLite (local dev) |
| Auth | Session-based, Werkzeug password hashing |
| Image handling | Pillow, NumPy |
| Deployment | Render |

## 📁 Project Structure

```
CNN-Pet-Classifier/
├── app.py                  # Flask app, routes, auth, inference endpoint
├── models.py                # SQLAlchemy models: User, Feedback
├── train.py                  # Fine-tuning logic (loads corrected feedback, retrains, re-exports TFLite)
├── Oxford-IIIT.keras         # Full Keras model (used as the base for fine-tuning)
├── cat_dog_model.tflite      # Quantized/converted model actually served at inference time
├── templates/                 # Jinja2 HTML templates
├── static/                    # CSS/JS/assets
├── requirements.txt
└── runtime.txt
```

## 🔄 How the feedback → fine-tuning loop works

1. A logged-in user uploads a photo → the TFLite model predicts a breed → the prediction and image are logged to the `Feedback` table
2. If the prediction is wrong, the user can submit the correct breed via the `/correct/<id>` flow
3. An admin reviews accumulated corrected feedback from `/admin`
4. Hitting **Fine-tune** (`/admin/fine_tune`) triggers `train.py`:
   - Builds a small training set from the corrected feedback images
   - Loads the full Keras model and fine-tunes it for a few epochs at a low learning rate
   - Re-exports both a new `.keras` file and a new `.tflite` file
5. On success, the app swaps in the new model files and reloads the TFLite interpreter **in place** — no restart or redeploy needed
6. Used feedback entries are cleared so the same corrections aren't retrained on twice

This closes the loop between real user corrections and model improvement without any manual retraining pipeline.

## 🚀 Running locally

```bash
git clone https://github.com/ChSumedh/CNN-Pet-Classifier.git
cd CNN-Pet-Classifier
pip install -r requirements.txt
python app.py
```

By default it falls back to a local SQLite database if `DATABASE_URL` isn't set — no extra setup needed to run locally.

### Environment variables

| Variable | Purpose | Required |
|---|---|---|
| `SECRET_KEY` | Signs Flask session cookies | Recommended (falls back to a dev key) |
| `DATABASE_URL` | Postgres connection string | Optional — defaults to local SQLite |

On first run, an admin account is auto-created (`admin` / `changeme123`) — **change this password immediately** if deploying publicly.

## ☁️ Deployment notes

Deployed on Render with a managed PostgreSQL database (SQLite's file-based storage doesn't persist across container restarts on most PaaS hosts, so production uses Postgres instead). TensorFlow is lazy-imported inside the fine-tuning route rather than at module load, keeping the base app lightweight to boot and avoiding unnecessary memory overhead on requests that don't need it.

## 🔐 Security notes

- Passwords are hashed with Werkzeug's `generate_password_hash` (salted, never stored in plaintext)
- Session-based auth gates the prediction, admin, and fine-tuning routes
- Admin-only actions (`/admin/*`) check `isAdmin` server-side before executing

## 📈 Possible future improvements

- CSRF protection on forms (Flask-WTF)
- Rate limiting on login and prediction endpoints
- Move fine-tuning off the request thread into a background job/worker so training doesn't block the web process
- Confidence scores displayed alongside predictions
- Expand beyond the Oxford-IIIT 37-class set

---

Built as a demonstration of end-to-end ML application development: model training, deployment, live inference, and a real feedback-driven retraining loop — not just a static classifier.
