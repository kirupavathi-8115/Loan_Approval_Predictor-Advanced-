# Customer Persona Segmentation

An unsupervised machine learning project to segment customers into distinct personas based on behavior, featuring a FastAPI backend and a Streamlit dashboard.

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the Model:
   *(The synthetic dataset is located in `data/raw/customers.csv`)*
   ```bash
   python ml/train.py
   ```
   This will output `segmenter.joblib`, `persona_map.json`, and `cluster_profiles.csv` in the `artifacts/` folder.

## Running the Apps

Start the FastAPI server and the Streamlit dashboard:

**Option 1: Using the PowerShell script**
```powershell
.\run_apps.ps1
```

**Option 2: Manually**
Terminal 1 (API):
```bash
uvicorn api.main:app --reload
```
Terminal 2 (Dashboard):
```bash
streamlit run app/streamlit_app.py
```

## Architecture
- **ML Pipeline**: Uses `StandardScaler` and `KMeans` via a `scikit-learn` pipeline.
- **Backend**: FastAPI serving the model with Pydantic for input validation.
- **Frontend**: Streamlit dashboard with custom CSS for rich aesthetics and interactivity.
