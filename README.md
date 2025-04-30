# 👗 Fashion Recommendation System

This is a content-based fashion recommendation system built using **TensorFlow**, **ResNet50**, and **Streamlit/Flask**. It suggests visually similar fashion items when a user uploads a clothing image. Ideal for e-commerce, personalization, and visual search applications.

---

## 📂 Project Structure

- `main.py` – Streamlit app for interactive image-based recommendations  
- `test.py` – Command-line based test of the recommendation system  
- `api.py` – Flask API that accepts image uploads and returns recommended items  
- `embeddings.pkl` – Precomputed image feature vectors (stored externally)  
- `filenames.pkl` – Filepaths of catalog images  
- `images/` – Fashion image catalog  
- `uploads/` – Temporary folder to handle uploaded images  
- `sample/` – Sample input images for testing  

---

## 🚀 How It Works

1. User uploads a fashion item image (like a shirt, bag, etc.).  
2. The system extracts features using a pre-trained **ResNet50** model.  
3. It compares the features with precomputed embeddings.  
4. Returns 5 most visually similar images.  

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` isn't available, install manually:
```bash
pip install streamlit flask pillow scikit-learn numpy tensorflow
```

---

## 📦 Download `embeddings.pkl`

The file `embeddings.pkl` is too large to be stored on GitHub.  
Please download it manually from Google Drive:

🔗 [Download embeddings.pkl](https://drive.google.com/uc?export=download&id=1_gRp2XopLC2d8VJcUkEPXiK2_kUXkOth)

After downloading, place it in the **root directory** of the project.

---

## 🧪 Running the Project

### Option 1: Use the Streamlit Interface (GUI)

```bash
streamlit run main.py
```

### Option 2: Use the Flask API

```bash
python api.py
```

- Visit `http://localhost:5500` to verify the API is running.  
- Send a POST request to `/recommend` with an image file.

---

## 📷 Sample Usage

- Upload an image like `sample/saree.jpeg`  
- Get 5 visually similar fashion items displayed  
- Try with your own fashion photos!

---

## 📝 License

This project is open-source and free to use for educational and non-commercial purposes.

---

## 🤝 Contributing

Pull requests and feedback are welcome! If you find a bug or want to suggest a feature, feel free to open an issue.
