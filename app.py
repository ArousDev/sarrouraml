import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st_lottie = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from xgboost import XGBClassifier


DATA_PATH = "Crop_recommendationV2-1.csv"


def load_lottieurl(url: str):
    """Load a Lottie animation from a URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def optimize_dbscan_parameters(X_scaled, eps_range=None, min_samples_range=None):
    """
    Teste différents paramètres DBSCAN pour trouver les meilleurs.
    
    Args:
        X_scaled: Données scalées
        eps_range: Liste de valeurs eps à tester (si None, calcule automatiquement)
        min_samples_range: Liste de valeurs min_samples à tester (si None, utilise [5, 10, 15, 20])
    
    Returns:
        DataFrame avec les résultats de chaque combinaison de paramètres
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Calculer automatiquement eps_range si non fourni
    if eps_range is None:
        # Utiliser la méthode k-distance pour suggérer eps
        k = 4  # k = min_samples - 1
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)[::-1]
        
        # Suggérer des valeurs eps basées sur les distances k-nearest
        eps_min = np.percentile(k_distances_sorted, 5)
        eps_max = np.percentile(k_distances_sorted, 95)
        eps_range = np.linspace(eps_min, eps_max, 20).round(3)
    
    if min_samples_range is None:
        min_samples_range = [5, 10, 15, 20, 25]
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = int(np.sum(clusters == -1))
            noise_pct = (n_noise / len(clusters)) * 100
            
            # Calculer silhouette score si possible
            mask = clusters != -1
            silhouette = None
            if len(np.unique(clusters[mask])) > 1 and n_clusters > 1:
                try:
                    silhouette = silhouette_score(X_scaled[mask], clusters[mask])
                except:
                    silhouette = None
            
            # Score composite pour évaluer la qualité
            # On veut: beaucoup de clusters, peu de bruit, bon silhouette score
            if n_clusters > 0:
                score = (n_clusters * 0.3) - (noise_pct * 0.2)
                if silhouette is not None:
                    score += (silhouette * 0.5)
            else:
                score = -100  # Pénalité si aucun cluster
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': round(noise_pct, 2),
                'silhouette': round(silhouette, 4) if silhouette is not None else None,
                'score': round(score, 4)
            })
    
    return pd.DataFrame(results)


@st.cache_resource
def perform_dbscan_clustering(df_raw: pd.DataFrame, eps=2.0, min_samples=10):
    """DBSCAN clustering avec paramètres configurables."""
    numerical_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "soil_moisture",
        "sunlight_exposure",
        "wind_speed",
    ]

    X = df_raw[numerical_features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN with specified parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise = int(list(clusters).count(-1))

    stats = {
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_pct": float(n_noise / len(clusters) * 100),
        "cluster_labels": unique_clusters.tolist(),
        "eps": eps,
        "min_samples": min_samples,
    }

    # Silhouette score excluding noise, if possible
    mask = clusters != -1
    if len(np.unique(clusters[mask])) > 1:
        stats["silhouette"] = float(
            silhouette_score(X_scaled[mask], clusters[mask])
        )
    else:
        stats["silhouette"] = None

    return clusters, X_scaled, stats, dbscan


def visualize_dbscan_results(clusters, X_scaled):
    """2D visualization of DBSCAN clustering results with PCA."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: clusters in PCA space
    ax = axes[0]
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("DBSCAN Clustering Results")

    # Mark noise points
    noise_mask = clusters == -1
    if any(noise_mask):
        ax.scatter(
            X_pca[noise_mask, 0],
            X_pca[noise_mask, 1],
            c="red",
            marker="x",
            s=5,
            label="Noise",
        )
        ax.legend()

    # Plot 2: cluster size distribution
    ax2 = axes[1]
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    colors = ["red" if idx == -1 else "skyblue" for idx in cluster_counts.index]
    ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of Points")
    ax2.set_title("Cluster Size Distribution")

    plt.tight_layout()
    return fig


def visualize_clustering_model(model_name, model, X_scaled_train, input_scaled, cluster_label):
    """
    Visualise un modèle de clustering avec les données d'entraînement et le point d'entrée.
    
    Args:
        model_name: Nom du modèle
        model: Modèle de clustering entraîné
        X_scaled_train: Données d'entraînement scalées
        input_scaled: Point d'entrée scalé (1 sample)
        cluster_label: Label du cluster assigné au point d'entrée
    """
    # Obtenir les labels de cluster pour toutes les données d'entraînement
    if isinstance(model, DBSCAN):
        # DBSCAN: utiliser les labels déjà calculés lors de l'entraînement
        train_clusters = model.labels_
    elif isinstance(model, KMeans):
        # KMeans: utiliser les labels déjà calculés
        train_clusters = model.labels_
    elif isinstance(model, AgglomerativeClustering):
        # Agglomerative: utiliser les labels déjà calculés
        train_clusters = model.labels_
    else:
        # Fallback: prédire pour toutes les données
        train_clusters = model.predict(X_scaled_train)
    
    # Réduire à 2D avec PCA pour la visualisation
    pca = PCA(n_components=2)
    X_pca_train = pca.fit_transform(X_scaled_train)
    input_pca = pca.transform(input_scaled)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Obtenir les clusters uniques
    unique_clusters = np.unique(train_clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    # Utiliser une colormap pour les clusters (différente pour DBSCAN)
    if isinstance(model, DBSCAN):
        colors_map = plt.cm.get_cmap('Set3', max(n_clusters, 1))
    else:
        colors_map = plt.cm.get_cmap('tab20', max(n_clusters, 1))
    
    # Tracer chaque cluster
    # Pour DBSCAN, afficher d'abord les clusters, puis le bruit avec une meilleure visualisation
    if isinstance(model, DBSCAN):
        # Utiliser une colormap plus vive pour DBSCAN
        dbscan_colors = plt.cm.get_cmap('Set3', max(n_clusters, 1))
        
        # Afficher d'abord les clusters avec des couleurs vives et distinctes
        cluster_sizes = {}
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id != -1:
                mask = train_clusters == cluster_id
                cluster_sizes[cluster_id] = np.sum(mask)
                # Utiliser des couleurs plus vives
                color = dbscan_colors(i % dbscan_colors.N)
                # Taille variable selon la densité du cluster
                point_size = max(80, min(150, cluster_sizes[cluster_id] // 5))
                
                ax.scatter(
                    X_pca_train[mask, 0],
                    X_pca_train[mask, 1],
                    c=[color],
                    s=point_size,
                    alpha=0.8,
                    label=f'Cluster {int(cluster_id)} ({cluster_sizes[cluster_id]} points)',
                    edgecolors='white',
                    linewidths=1.2,
                    zorder=5
                )
        
        # Afficher le bruit en dernier avec un style distinct
        noise_mask = train_clusters == -1
        if any(noise_mask):
            n_noise = np.sum(noise_mask)
            ax.scatter(
                X_pca_train[noise_mask, 0], 
                X_pca_train[noise_mask, 1],
                c='#d3d3d3',
                marker='x',
                s=60,
                alpha=0.6,
                label=f'Noise Points ({n_noise} points)',
                linewidths=2,
                zorder=4
            )
    else:
        # Pour les autres modèles (KMeans, Agglomerative)
        for i, cluster_id in enumerate(unique_clusters):
            mask = train_clusters == cluster_id
            color = colors_map(i % colors_map.N)
            ax.scatter(
                X_pca_train[mask, 0],
                X_pca_train[mask, 1],
                c=[color],
                s=50,
                alpha=0.6,
                label=f'Cluster {int(cluster_id)}',
                edgecolors='black',
                linewidths=0.3
            )
    
    # Tracer le point d'entrée avec un marqueur spécial (toujours en noir)
    if cluster_label == -1:
        input_color = 'black'
        input_marker = 'X'
        input_label = 'Your Input (Noise)'
        input_size = 300
    else:
        input_color = 'black'  # Toujours en noir pour être facilement identifiable
        input_marker = '*'  # Utiliser '*' au lieu de '★'
        input_label = f'Your Input (Cluster {int(cluster_label)})'
        input_size = 500
    
    ax.scatter(
        input_pca[0, 0],
        input_pca[0, 1],
        c=input_color,
        marker=input_marker,
        s=input_size,
        edgecolors='black',
        linewidths=2.5,
        label=input_label,
        zorder=10  # Au-dessus des autres points
    )
    
    # Centres de cluster pour KMeans
    if isinstance(model, KMeans):
        centers_scaled = model.cluster_centers_
        centers_pca = pca.transform(centers_scaled)
        # Tracer chaque centre avec la couleur de son cluster correspondant
        for center_idx in range(len(centers_pca)):
            # Trouver l'index du cluster dans unique_clusters
            cluster_id = center_idx  # Pour KMeans, les centres sont dans l'ordre 0, 1, 2, ...
            cluster_idx = np.where(unique_clusters == cluster_id)[0]
            if len(cluster_idx) > 0:
                center_color = colors_map(cluster_idx[0] % colors_map.N)
            else:
                center_color = 'black'
            
            # Ajouter le label seulement pour le premier centre pour éviter la répétition dans la légende
            ax.scatter(
                centers_pca[center_idx, 0],
                centers_pca[center_idx, 1],
                c=[center_color],
                marker='*',
                s=400,
                edgecolors='black',
                linewidths=2,
                label='Cluster Centers' if center_idx == 0 else '',
                zorder=9
            )
    
    ax.set_xlabel('PCA Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PCA Component 2', fontsize=12, fontweight='bold')
    
    # Titre adapté selon le modèle avec statistiques pour DBSCAN
    if isinstance(model, DBSCAN):
        n_clusters_found = len([c for c in unique_clusters if c != -1])
        n_noise = np.sum(train_clusters == -1)
        title = f'{model_name} Clustering Visualization\n'
        title += f'Found {n_clusters_found} clusters, {n_noise} noise points\n'
        if cluster_label == -1:
            title += 'Your Input: Noise Point (Outlier)'
        else:
            title += f'Your Input: Cluster {int(cluster_label)}'
    else:
        title = f'{model_name} Clustering Visualization\nInput assigned to Cluster {int(cluster_label)}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Améliorer l'apparence générale
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    return fig


def render_colorful_table(df: pd.DataFrame, table_id: str = ""):
    """Render a colorful, slightly animated table using HTML/CSS."""
    if df is None or df.empty:
        return

    html = df.to_html(
        index=False,
        classes=f"agri-table agri-table-{table_id}",
        border=0,
        justify="center",
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def train_models(df: pd.DataFrame):
    # Split features / target
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    # For XGBoost we must encode string labels into integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encoded version of y for XGBoost
    y_train_enc = label_encoder.transform(y_train)

    # Supervised models (classification)
    models_cls = {}

    models_cls["SVM"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, kernel="rbf", random_state=42)),
        ]
    )

    models_cls["KNN"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    models_cls["Random Forest"] = RandomForestClassifier(
        n_estimators=150, random_state=42
    )

    models_cls["Decision Tree"] = DecisionTreeClassifier(
        max_depth=10, random_state=42
    )

    models_cls["Linear Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeClassifier(random_state=42)),
        ]
    )

    models_cls["Polynomial Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)),
        ]
    )

    models_cls["Logistic Regression (Binary)"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))),
        ]
    )

    models_cls["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    for name, model in models_cls.items():
        if name == "XGBoost":
            # XGBoost requires integer-encoded labels
            model.fit(X_train, y_train_enc)
        else:
            model.fit(X_train, y_train)

    # Unsupervised models (clustering)
    # We scale for clustering as well
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = len(np.unique(y))

    # Utiliser les paramètres DBSCAN fixes
    dbscan_eps = 4.343
    dbscan_min_samples = 5
    
    models_cluster = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "ACH": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
    }

    for model in models_cluster.values():
        if isinstance(model, (KMeans, AgglomerativeClustering, DBSCAN)):
            model.fit(X_scaled)

    return {
        "feature_cols": feature_cols,
        "scaler_cluster": scaler,
        "X_scaled_cluster": X_scaled,  # Données scalées pour la visualisation
        "label_encoder": label_encoder,
        "models_cls": models_cls,
        "models_cluster": models_cluster,
    }


def build_input_form(feature_cols, df: pd.DataFrame):
    st.subheader("📥 Enter the feature values")
    user_input = {}

    # Icons for some key features (fallback to a dot if not defined)
    feature_icons = {
        "N": "🧪",
        "P": "🧪",
        "K": "🧪",
        "temperature": "🌡️",
        "humidity": "💧",
        "ph": "⚗️",
        "rainfall": "🌧️",
        "soil_moisture": "🌱",
        "soil_type": "🪨",
        "sunlight_exposure": "☀️",
        "wind_speed": "🌬️",
        "co2_concentration": "🌫️",
        "organic_matter": "🍂",
        "irrigation_frequency": "🚿",
        "crop_density": "🌾",
        "pest_pressure": "🐛",
        "fertilizer_usage": "🧴",
        "growth_stage": "🌿",
        "urban_area_proximity": "🏙️",
        "water_source_type": "🚰",
        "frost_risk": "❄️",
        "water_usage_efficiency": "💧",
    }

    # Use dataset statistics to give nicer default values and ranges
    desc = df[feature_cols].describe()

    # Use 2 columns instead of 3 for better alignment with longer feature names
    rows = []
    for i in range(0, len(feature_cols), 2):
        rows.append(feature_cols[i : i + 2])

    for row in rows:
        cols = st.columns(len(row))
        for col_name, col_container in zip(row, cols):
            icon = feature_icons.get(col_name, "•")

            # sensible defaults: median; range: min / max from data
            median_val = float(desc.loc["50%", col_name])
            min_val = float(desc.loc["min", col_name])
            max_val = float(desc.loc["max", col_name])

            with col_container:
                # Replace underscores with spaces and capitalize for better display
                display_name = col_name.replace("_", " ").title()
                st.markdown(
                    f"<div style='font-weight:600; margin-bottom:0.3rem; white-space:nowrap;'>{icon} {display_name}</div>",
                    unsafe_allow_html=True,
                )
                user_input[col_name] = st.number_input(
                    "",
                    value=median_val,
                    min_value=min_val,
                    max_value=max_val,
                    step=(max_val - min_val) / 100.0 if max_val > min_val else 0.1,
                    format="%.4f",
                    key=f"input_{col_name}",
                    label_visibility="collapsed",
                )

    return pd.DataFrame([user_input])


def get_crop_recommendations(input_df, models_cls, label_encoder, top_n=3):
    """
    Système de recommandation qui agrège les résultats de tous les modèles
    pour recommander les meilleures cultures à planter.
    
    Args:
        input_df: DataFrame avec les données d'entrée
        models_cls: Dictionnaire des modèles de classification
        label_encoder: LabelEncoder pour XGBoost
        top_n: Nombre de recommandations à retourner
    
    Returns:
        DataFrame avec les recommandations et leurs scores de confiance
    """
    all_predictions = []
    all_probabilities = {}
    
    # Obtenir les prédictions et probabilités de tous les modèles
    for name, model in models_cls.items():
        try:
            if name == "XGBoost":
                # XGBoost: décoder les labels
                encoded_pred = model.predict(input_df)[0]
                pred = label_encoder.inverse_transform([int(encoded_pred)])[0]
                # Obtenir les probabilités pour toutes les classes
                proba = model.predict_proba(input_df)[0]
                classes = label_encoder.classes_
            else:
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                classes = model.classes_
            
            all_predictions.append(pred)
            
            # Stocker les probabilités pour chaque classe
            for i, crop in enumerate(classes):
                if crop not in all_probabilities:
                    all_probabilities[crop] = []
                all_probabilities[crop].append(float(proba[i]))
                
        except Exception as e:
            continue
    
    # Calculer le score agrégé pour chaque culture
    recommendations = []
    for crop, probs in all_probabilities.items():
        # Score moyen de confiance
        avg_confidence = np.mean(probs)
        # Nombre de modèles qui recommandent cette culture
        vote_count = all_predictions.count(crop)
        # Score final = moyenne des probabilités * poids du vote
        final_score = avg_confidence * (1 + vote_count * 0.2)
        
        recommendations.append({
            "Crop": crop,
            "Average Confidence": round(avg_confidence, 4),
            "Vote Count": vote_count,
            "Final Score": round(final_score, 4)
        })
    
    # Trier par score final décroissant
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values("Final Score", ascending=False)
    
    # Retourner les top N recommandations
    return recommendations_df.head(top_n), all_predictions


def landing_page():
    """Beautiful landing page with introduction and navigation button."""
    st.set_page_config(
        page_title="Smart Agriculture – Crop Recommendation",
        page_icon="🌾",
        layout="wide",
    )
    
    # Custom CSS for landing page
    st.markdown(
        """
        <style>
        /* Animated gradient background */
        .stApp {
            background: linear-gradient(-45deg, #fdfbfb, #e0f7fa, #f1f8e9, #fff3e0, #f3e5f5, #e8f5e9);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Landing page hero section */
        .landing-hero {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 3rem 2rem;
            text-align: center;
        }
        
        .landing-title {
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
            animation: titleFloat 3s ease-in-out infinite;
            text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        
        @keyframes titleFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .landing-subtitle {
            font-size: 1.8rem;
            color: #4a5568;
            margin-bottom: 3rem;
            max-width: 800px;
            line-height: 1.8;
            animation: fadeInUp 1s ease-out;
        }
        
        /* Floating animated plants and seeds */
        .floating-icons {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 1;
            overflow: hidden;
        }
        
        .floating-icon {
            position: absolute;
            font-size: 3rem;
            opacity: 0.7;
            animation: floatAround 20s infinite ease-in-out;
        }
        
        .floating-icon:nth-child(1) {
            top: 10%;
            left: 10%;
            animation-delay: 0s;
            animation-duration: 15s;
        }
        
        .floating-icon:nth-child(2) {
            top: 20%;
            right: 15%;
            animation-delay: 2s;
            animation-duration: 18s;
        }
        
        .floating-icon:nth-child(3) {
            bottom: 30%;
            left: 20%;
            animation-delay: 4s;
            animation-duration: 22s;
        }
        
        .floating-icon:nth-child(4) {
            top: 50%;
            right: 10%;
            animation-delay: 1s;
            animation-duration: 16s;
        }
        
        .floating-icon:nth-child(5) {
            bottom: 20%;
            right: 25%;
            animation-delay: 3s;
            animation-duration: 19s;
        }
        
        .floating-icon:nth-child(6) {
            top: 70%;
            left: 15%;
            animation-delay: 5s;
            animation-duration: 21s;
        }
        
        .floating-icon:nth-child(7) {
            top: 15%;
            left: 50%;
            animation-delay: 1.5s;
            animation-duration: 17s;
        }
        
        .floating-icon:nth-child(8) {
            bottom: 15%;
            left: 40%;
            animation-delay: 2.5s;
            animation-duration: 20s;
        }
        
        .floating-icon:nth-child(9) {
            top: 60%;
            right: 30%;
            animation-delay: 0.5s;
            animation-duration: 14s;
        }
        
        .floating-icon:nth-child(10) {
            bottom: 40%;
            right: 20%;
            animation-delay: 3.5s;
            animation-duration: 23s;
        }
        
        @keyframes floatAround {
            0% {
                transform: translate(0, 0) rotate(0deg) scale(1);
            }
            25% {
                transform: translate(30px, -50px) rotate(90deg) scale(1.1);
            }
            50% {
                transform: translate(-20px, -80px) rotate(180deg) scale(0.9);
            }
            75% {
                transform: translate(-40px, -30px) rotate(270deg) scale(1.05);
            }
            100% {
                transform: translate(0, 0) rotate(360deg) scale(1);
            }
        }
        
        .landing-hero {
            position: relative;
            z-index: 10;
        }
        
        .landing-button-container {
            margin-top: 4rem;
            animation: fadeInUp 1.2s ease-out;
            position: relative;
            z-index: 10;
        }
        
        .landing-button-wrapper {
            display: flex;
            justify-content: center;
            margin-top: 3rem;
        }
        
        .landing-button-custom {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 1.2rem 3.5rem;
            font-size: 1.3rem;
            font-weight: 700;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            text-decoration: none;
            display: inline-block;
        }
        
        .landing-button-custom::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }
        
        .landing-button-custom:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
            color: white !important;
        }
        
        .landing-button-custom:hover::before {
            left: 100%;
        }
        
        .landing-button-custom:active {
            transform: translateY(-1px) scale(1.02);
        }
        
        /* Style pour le bouton Streamlit */
        .landing-page .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 1.2rem 3.5rem !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 50px !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
        }
        
        .landing-page .stButton > button:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
            background: linear-gradient(135deg, #7c8ff5 0%, #8b5fbf 100%) !important;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 30px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }
        
        /* Hide default Streamlit elements on landing page */
        .landing-page [data-testid="stSidebar"] {
            display: none;
        }
        
        .landing-page [data-testid="stHeader"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Landing page content
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)
    
    # Floating animated icons (plants and seeds)
    st.markdown(
        """
        <div class="floating-icons">
            <div class="floating-icon">🌾</div>
            <div class="floating-icon">🌱</div>
            <div class="floating-icon">🌿</div>
            <div class="floating-icon">🌽</div>
            <div class="floating-icon">🌻</div>
            <div class="floating-icon">🌷</div>
            <div class="floating-icon">🌰</div>
            <div class="floating-icon">🌾</div>
            <div class="floating-icon">🌱</div>
            <div class="floating-icon">🌿</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
     st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-title">🌾 Smart Agriculture AI Lab</div>
            <div class="landing-subtitle">
              Discover the future of smart agriculture with our advanced AI platform.
<br><br>
Get personalized crop recommendations based on your environmental data using 8 advanced machine learning models .
<br><br>
Explore interactive visualizations and receive intelligent recommendations with confidence scores to optimize your agricultural decisions.
          
         
        """,
        unsafe_allow_html=True,
    )
    
    # Button to navigate to main app
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_clicked = st.button("🚀 Commencer l'Exploration", type="primary", use_container_width=True, key="landing_button")
        if button_clicked:
            st.session_state['show_main_app'] = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def main_app():
    """Main application interface."""
    st.set_page_config(
        page_title="Smart Agriculture – Crop Recommendation",
        page_icon="🌾",
        layout="wide",
    )

    # Custom CSS for a colorful, modern look with animated background
    st.markdown(
        """
        <style>
        /* Animated gradient background with floating particles */
        .stApp {
            background: linear-gradient(-45deg, #fdfbfb, #e0f7fa, #f1f8e9, #fff3e0, #f3e5f5, #e8f5e9);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        @keyframes gradientShift {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        .main {
            background: transparent !important;
        }
        /* Modern Sidebar - Vibrant Gradient Design */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, 
                rgba(139, 92, 246, 0.95) 0%, 
                rgba(99, 102, 241, 0.95) 25%,
                rgba(59, 130, 246, 0.95) 50%,
                rgba(99, 102, 241, 0.95) 75%,
                rgba(139, 92, 246, 0.95) 100%) !important;
            background-size: 100% 400%;
            animation: sidebarGradientFlow 12s ease infinite;
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-right: 2px solid rgba(255, 255, 255, 0.2);
            box-shadow: 4px 0 32px rgba(139, 92, 246, 0.3);
            position: relative;
            overflow: hidden;
        }
        @keyframes sidebarGradientFlow {
            0%, 100% {
                background-position: 0% 0%;
            }
            50% {
                background-position: 0% 100%;
            }
        }
        [data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(147, 197, 253, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(196, 181, 253, 0.1) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
            animation: sidebarShimmer 8s ease-in-out infinite;
        }
        @keyframes sidebarShimmer {
            0%, 100% {
                opacity: 0.6;
            }
            50% {
                opacity: 1;
            }
        }
        [data-testid="stSidebar"] > * {
            position: relative;
            z-index: 1;
        }
        /* Sidebar Content Styling */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stMultiSelect {
            margin-bottom: 1.5rem;
        }
        /* Modern Sidebar Headers - Vibrant Colors */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.75rem;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1.1rem;
            color: #e0e7ff !important;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
        }
        /* Sidebar Text */
        [data-testid="stSidebar"] .stMarkdown {
            color: #f0f9ff !important;
            line-height: 1.6;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        }
        /* Sidebar Labels - Modern Style */
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            display: block;
            text-shadow: 0 1px 5px rgba(0, 0, 0, 0.3);
        }
        /* Modern Input Fields - Vibrant Theme */
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] [data-baseweb="select"]:hover,
        [data-testid="stSidebar"] [data-baseweb="base-input"]:hover,
        [data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.15) 100%) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        [data-testid="stSidebar"] [data-baseweb="select"]:focus,
        [data-testid="stSidebar"] [data-baseweb="base-input"]:focus,
        [data-testid="stSidebar"] [data-baseweb="select"] > div:focus,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div:focus {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.3) 0%, rgba(255, 255, 255, 0.2) 100%) !important;
            border-color: rgba(255, 255, 255, 0.7) !important;
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.2), 0 4px 20px rgba(255, 255, 255, 0.3);
        }
        /* Remove white backgrounds from multiselect containers */
        [data-testid="stSidebar"] .stMultiSelect > div,
        [data-testid="stSidebar"] .stSelectbox > div,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
            color: #ffffff !important;
        }
        /* Multiselect container styling */
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="listbox"],
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="option"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.95) 0%, rgba(99, 102, 241, 0.95) 100%) !important;
            color: #ffffff !important;
            backdrop-filter: blur(20px);
        }
        [data-testid="stSidebar"] [data-baseweb="popover"] [role="option"]:hover {
            background: linear-gradient(135deg, rgba(167, 139, 250, 0.9) 0%, rgba(129, 140, 248, 0.9) 100%) !important;
            transform: translateX(4px);
            transition: all 0.2s ease;
        }
        /* Modern Tag/Chip Styling - Vibrant Gradient */
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%) !important;
            color: #6366f1 !important;
            border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 14px !important;
            padding: 7px 14px !important;
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3), 0 2px 8px rgba(139, 92, 246, 0.4);
            transition: all 0.3s ease;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"]:hover {
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%) !important;
            box-shadow: 0 6px 20px rgba(255, 255, 255, 0.4), 0 4px 12px rgba(139, 92, 246, 0.5);
            transform: translateY(-2px) scale(1.05);
        }
        /* All tag variations */
        [data-testid="stSidebar"] span[style*="background-color: rgb(239, 68, 68)"],
        [data-testid="stSidebar"] span[style*="background-color: #ef4444"],
        [data-testid="stSidebar"] .stMultiSelect [role="button"],
        [data-testid="stSidebar"] [data-baseweb="tag"] span,
        [data-testid="stSidebar"] div[data-baseweb="tag"] {
            background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%) !important;
            color: #6366f1 !important;
            border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 14px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"]:hover,
        [data-testid="stSidebar"] .stMultiSelect [role="button"]:hover {
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%) !important;
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 6px 20px rgba(255, 255, 255, 0.4);
        }
        /* Tag close button */
        [data-testid="stSidebar"] [data-baseweb="tag"] svg {
            fill: #6366f1 !important;
            opacity: 0.7;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] svg:hover {
            opacity: 1;
            fill: #4f46e5 !important;
        }
        /* Dropdown menu styling */
        [data-testid="stSidebar"] [data-baseweb="popover"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.98) 0%, rgba(99, 102, 241, 0.98) 100%) !important;
            backdrop-filter: blur(20px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            box-shadow: 0 12px 40px rgba(139, 92, 246, 0.5), 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        /* Remove all white backgrounds from sidebar elements */
        [data-testid="stSidebar"] div[style*="background-color: white"],
        [data-testid="stSidebar"] div[style*="background-color: #fff"],
        [data-testid="stSidebar"] div[style*="background-color: rgb(255, 255, 255)"],
        [data-testid="stSidebar"] [style*="background-color: white"],
        [data-testid="stSidebar"] [style*="background-color: #fff"],
        [data-testid="stSidebar"] [style*="background-color: rgb(255, 255, 255)"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
        }
        /* Multiselect and selectbox container backgrounds */
        [data-testid="stSidebar"] .stMultiSelect,
        [data-testid="stSidebar"] .stSelectbox {
            background: transparent !important;
        }
        [data-testid="stSidebar"] .stMultiSelect > div > div,
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
        }
        /* Scrollbar styling for sidebar - Vibrant */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 8px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.6) 0%, rgba(255, 255, 255, 0.4) 100%);
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.6) 100%);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }
        [data-testid="main"] {
            background: transparent !important;
        }
        /* Ensure all content is visible */
        .stMarkdown, .stContainer, .element-container {
            visibility: visible !important;
            opacity: 1 !important;
        }
        .agri-header {
            padding: 2.5rem 3rem;
            border-radius: 24px;
            background: linear-gradient(90deg, #00b894, #55efc4, #74b9ff, #a29bfe, #fd79a8, #fdcb6e, #00b894);
            background-size: 300% 100%;
            animation: headerGradient 8s ease infinite;
            color: white;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.15);
            margin: 0 auto 2rem auto;
            position: relative;
            overflow: hidden;
            max-width: 90%;
            text-align: center;
        }
        .agri-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 30px 30px;
            animation: sparkle 10s linear infinite;
            pointer-events: none;
        }
        @keyframes headerGradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        @keyframes sparkle {
            0% {
                transform: translate(0, 0) rotate(0deg);
            }
            100% {
                transform: translate(50px, 50px) rotate(360deg);
            }
        }
        .agri-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
            animation: titlePulse 3s ease-in-out infinite;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            text-align: center;
        }
        @keyframes titlePulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.02);
            }
        }
        .agri-subtitle {
            font-size: 1.3rem;
            opacity: 0.95;
            position: relative;
            z-index: 1;
            text-align: center;
            line-height: 1.6;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            border: none;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #7c8ff5 0%, #8b5fbf 100%);
        }
        .stButton>button:hover::before {
            left: 100%;
        }
        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
        }
        .agri-section {
            padding: 1rem 1.5rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
        /* Feature input styling */
        .stNumberInput label p {
            color: #00695c !important;
            font-weight: 600;
        }
        .stNumberInput input {
            border-radius: 12px !important;
            border: 1px solid #cfd8dc !important;
            width: 100% !important;
        }
        /* Ensure feature labels are properly aligned */
        [data-testid="column"] {
            padding: 0.5rem !important;
        }
        /* Better spacing for feature inputs */
        .element-container {
            margin-bottom: 0.8rem !important;
        }
        /* Ensure consistent column widths */
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: stretch;
        }
        /* Better spacing for results section */
        .stSubheader {
            margin-top: 1.5rem !important;
            margin-bottom: 0.8rem !important;
        }
        /* Ensure tables don't overflow */
        .agri-table {
            max-width: 100%;
            overflow-x: auto;
        }
        /* Colorful animated tables */
        .agri-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 14px;
            overflow: hidden;
            background: linear-gradient(135deg, #ffffff, #e8f5e9);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            animation: fadeInUp 0.6s ease-out;
        }
        .agri-table thead {
            background: linear-gradient(90deg, #00b894, #55efc4);
            color: #ffffff;
            font-weight: 700;
        }
        .agri-table th, .agri-table td {
            padding: 0.7rem 0.9rem;
            font-size: 0.95rem;
        }
        .agri-table tbody tr:nth-child(even) {
            background-color: rgba(232, 245, 233, 0.85);
        }
        .agri-table tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }
        .agri-table tbody tr:hover {
            background: linear-gradient(90deg, #e3f2fd, #f1f8e9);
            transform: translateY(-1px);
            transition: all 0.2s ease;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 12px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Show initial message to confirm page is loading
    st.markdown("")  # Small spacer
    
    # Hero header with animation
    try:
        lottie_agri = load_lottieurl(
            "https://lottie.host/72afb34f-9c97-4c0e-9af3-df2480b5e4cb/cn3nIhB9mS.json"
        )
    except Exception:
        lottie_agri = None

    # Centered header
    st.markdown(
        """
        <div style="display: flex; justify-content: center; width: 100%;">
            <div class="agri-header">
                <div class="agri-title">🌾 Smart Agriculture AI Lab</div>
                <div class="agri-subtitle">
                    Explore your farming data with <b>DBSCAN, KMeans, ACH, SVM, KNN, Random Forest, Decision Tree, Linear Regression, Polynomial Regression, Logistic Regression, XGBoost</b> –
                    all in one interactive, colorful dashboard.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data and train models
    try:
        with st.spinner("Loading data and training models..."):
            df = load_data(DATA_PATH)
            model_bundle = train_models(df)

        feature_cols = model_bundle["feature_cols"]
        models_cls = model_bundle["models_cls"]
        models_cluster = model_bundle["models_cluster"]
        scaler_cluster = model_bundle["scaler_cluster"]
        X_scaled_cluster = model_bundle["X_scaled_cluster"]
        label_encoder = model_bundle["label_encoder"]
    except Exception as e:
        st.error(f"Error loading data or training models: {str(e)}")
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    st.sidebar.markdown("### Choose models to run")
    selected_supervised = st.sidebar.multiselect(
        "Supervised models",
        list(models_cls.keys()),
        default=list(models_cls.keys()),
    )
    selected_cluster = st.sidebar.multiselect(
        "Clustering models",
        list(models_cluster.keys()),
        default=list(models_cluster.keys()),
    )

    with st.container():
        st.markdown("### 📊 Dataset overview")
        with st.expander("Show / hide dataset preview"):
            st.write(df.head())
            st.write("Shape:", df.shape)


    # Layout: inputs first, then results below
    st.markdown("### 🤖 Crop prediction & clustering")
    
    # Input form section
    input_df = build_input_form(feature_cols, df)
    predict_button = st.button("🚀 Run models", type="primary")

    # Results section - appears below the form when button is clicked
    if predict_button:
        st.markdown("---")  # Separator line
        st.subheader("📌 Supervised models – Crop prediction")

        if selected_supervised:
            sup_results = []
            for name in selected_supervised:
                model = models_cls[name]
                # XGBoost model outputs encoded labels; decode them
                if name == "XGBoost":
                    encoded_pred = model.predict(input_df)[0]
                    pred = label_encoder.inverse_transform(
                        [int(encoded_pred)]
                    )[0]
                else:
                    pred = model.predict(input_df)[0]
                # Try to get probability if available
                try:
                    proba = np.max(model.predict_proba(input_df))
                except Exception:
                    proba = np.nan

                sup_results.append(
                    {
                        "Model": name,
                        "Predicted crop": str(pred),
                        "Confidence (max prob)": round(float(proba), 4)
                        if not np.isnan(proba)
                        else None,
                    }
                )

            render_colorful_table(pd.DataFrame(sup_results), table_id="supervised")
        else:
            st.info("Select at least one supervised model in the sidebar.")

        # Section de recommandation intelligente
        st.markdown("---")  # Separator line
        st.subheader("🌾 Recommandations de cultures - Système intelligent")
        
        if selected_supervised:
            try:
                recommendations_df, all_predictions = get_crop_recommendations(
                    input_df, 
                    {name: models_cls[name] for name in selected_supervised},
                    label_encoder,
                    top_n=5
                )
                
                # Afficher la recommandation principale
                if not recommendations_df.empty:
                    top_recommendation = recommendations_df.iloc[0]
                    st.success(
                        f"🎯 **Recommandation principale:** {top_recommendation['Crop']} "
                        f"(Confiance: {top_recommendation['Average Confidence']*100:.1f}%, "
                        f"Votes: {top_recommendation['Vote Count']}/{len(selected_supervised)})"
                    )
                    
                    # Afficher toutes les recommandations
                    st.markdown("#### 📊 Top recommandations (triées par score):")
                    
                    # Créer un DataFrame formaté pour l'affichage
                    display_df = recommendations_df.copy()
                    display_df['Average Confidence'] = (display_df['Average Confidence'] * 100).round(2).astype(str) + '%'
                    display_df = display_df.rename(columns={
                        'Crop': '🌾 Culture recommandée',
                        'Average Confidence': 'Confiance moyenne',
                        'Vote Count': 'Nombre de votes',
                        'Final Score': 'Score final'
                    })
                    
                    render_colorful_table(display_df, table_id="recommendations")
                    
                    # Afficher le consensus des modèles
                    st.markdown("#### 🤝 Consensus des modèles:")
                    consensus_text = ", ".join([f"{crop} ({all_predictions.count(crop)} votes)" 
                                               for crop in set(all_predictions)])
                    st.info(consensus_text)
                else:
                    st.warning("Impossible de générer des recommandations.")
            except Exception as e:
                st.error(f"Erreur lors de la génération des recommandations: {str(e)}")
        else:
            st.info("Sélectionnez au moins un modèle supervisé dans la barre latérale pour obtenir des recommandations.")

        st.markdown("---")  # Separator line
        st.subheader("📌 Clustering models – Group / cluster assignment")
        if selected_cluster:
            x_scaled_single = scaler_cluster.transform(input_df)

            for name in selected_cluster:
                model = models_cluster[name]
                
                # Obtenir le label du cluster pour le point d'entrée
                try:
                    if isinstance(model, DBSCAN):
                        # DBSCAN ne peut pas prédire directement pour de nouveaux points
                        # On trouve le cluster le plus proche en calculant les distances
                        from sklearn.neighbors import NearestNeighbors
                        # Trouver les points du dataset les plus proches
                        nn = NearestNeighbors(n_neighbors=min(model.min_samples, len(X_scaled_cluster)))
                        nn.fit(X_scaled_cluster)
                        distances, indices = nn.kneighbors(x_scaled_single)
                        
                        # Obtenir les labels des points les plus proches
                        nearest_labels = model.labels_[indices[0]]
                        # Si tous les voisins sont du bruit, le point est aussi du bruit
                        if all(label == -1 for label in nearest_labels):
                            cluster_label = -1
                        else:
                            # Prendre le cluster le plus fréquent parmi les voisins (en excluant le bruit)
                            non_noise_labels = [l for l in nearest_labels if l != -1]
                            if len(non_noise_labels) > 0:
                                from collections import Counter
                                cluster_label = Counter(non_noise_labels).most_common(1)[0][0]
                            else:
                                cluster_label = -1
                    elif isinstance(model, AgglomerativeClustering):
                        # AgglomerativeClustering n'a pas de méthode predict
                        # On trouve le cluster le plus proche en utilisant les distances
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1)
                        nn.fit(X_scaled_cluster)
                        distances, indices = nn.kneighbors(x_scaled_single)
                        # Utiliser le label du point le plus proche
                        cluster_label = model.labels_[indices[0][0]]
                    else:
                        # KMeans et autres modèles avec méthode predict
                        cluster_label = model.predict(x_scaled_single)[0]
                    
                    # Créer et afficher le graphique pour ce modèle
                    st.markdown(f"#### 🎯 {name} Clustering")
                    
                    fig = visualize_clustering_model(
                        name,
                        model,
                        X_scaled_cluster,
                        x_scaled_single,
                        cluster_label
                    )
                    st.pyplot(fig)
                    
                    # Afficher l'information du cluster assigné
                    if cluster_label == -1:
                        st.warning(f"⚠️ **Entry point assigned to:** Noise (Cluster -1)")
                        st.info("This point is considered as Noise (outlier) by DBSCAN.")
                    else:
                        st.success(f"✅ **Entry point assigned to:** Cluster {int(cluster_label)}")
                    
                    st.markdown("---")  # Séparateur entre les modèles
                    
                except Exception as e:
                    st.error(f"Erreur lors de la visualisation du modèle {name}: {str(e)}")
                    st.markdown("---")
        else:
            st.info("Select at least one clustering model in the sidebar.")


def main():
    """Main entry point - shows landing page or main app based on session state."""
    # Initialize session state for navigation
    if 'show_main_app' not in st.session_state:
        st.session_state['show_main_app'] = False
    
    # Show landing page or main app
    if not st.session_state['show_main_app']:
        landing_page()
    else:
        main_app()


if __name__ == "__main__":
    main()


