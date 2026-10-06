import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from features import FeatureEngineer

def main():
    print("Loading data...")
    # Go one level up since this script is in ml/
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "customers.csv"
    
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    features_cols = ["age", "income", "total_spend", "num_purchases", "recency_days", "web_visits_per_month"]
    
    # Drop rows with NaN if needed, though FeatureEngineer handles it
    df = df.dropna(subset=features_cols, how='all')
    X = df[features_cols]
    
    n_clusters = 3
    pipe = Pipeline([
        ("engineer", FeatureEngineer()),
        ("scale", StandardScaler()),
        ("km", KMeans(n_clusters=n_clusters, n_init=10, random_state=42)),
    ])
    
    print("Training pipeline...")
    pipe.fit(X)
    
    cluster_labels = pipe.predict(X)
    df["cluster"] = cluster_labels
    
    X_processed = pipe.named_steps["scale"].transform(pipe.named_steps["engineer"].transform(X))
    sil_score = silhouette_score(X_processed, cluster_labels)
    db_score = davies_bouldin_score(X_processed, cluster_labels)
    
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Davies-Bouldin Score: {db_score:.3f}")
    
    print("Profiling clusters...")
    profile = df.groupby("cluster")[features_cols].mean().round(2)
    profile["size"] = df.groupby("cluster").size()
    
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    profile.to_csv(artifacts_dir / "cluster_profiles.csv")
    
    persona_map = {}
    for cluster_id in range(n_clusters):
        row = profile.loc[cluster_id]
        
        # Simple heuristic mapping for the 3 personas
        if row["total_spend"] > profile["total_spend"].median() and row["recency_days"] < profile["recency_days"].median():
            name = "High-Value Loyalist"
            desc = "Spends heavily and buys frequently."
        elif row["recency_days"] > profile["recency_days"].median() and row["num_purchases"] < profile["num_purchases"].median():
            name = "At-Risk / Dormant"
            desc = "Hasn't purchased recently and rarely buys."
        else:
            name = "New / Occasional"
            desc = "Average spend and infrequent purchases."
            
        persona_map[str(cluster_id)] = {
            "name": name,
            "description": desc
        }
    
    with open(artifacts_dir / "persona_map.json", "w") as f:
        json.dump(persona_map, f, indent=4)
        
    print("Saving artifacts...")
    joblib.dump(pipe, artifacts_dir / "segmenter.joblib")
    print("Training complete.")

if __name__ == "__main__":
    main()
