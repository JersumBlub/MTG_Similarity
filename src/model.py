import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generates semantic embeddings for oracle text using a Transformer model.
    """
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} cards...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_feature_matrix(df, type_dummies, keyword_dummies, embeddings):
    """
    Combines numeric stats, categorical features, and text embeddings 
    with custom weighting for similarity.
    """
    # 1. Scale Numeric features (CMC and P/T)
    cmc_scaler = MinMaxScaler()
    cmc_scaled = pd.DataFrame(
        cmc_scaler.fit_transform(df[['cmc']].fillna(0)),
        index=df.index, columns=['cmc']
    )

    pt_features = ['p_main', 'p_side', 'p_var', 't_main', 't_side', 't_var']
    pt_scaler = MinMaxScaler()
    pt_scaled = pd.DataFrame(
        pt_scaler.fit_transform(df[pt_features].fillna(0)),
        index=df.index, columns=pt_features
    )

    # 2. Normalize and structure embeddings
    embedding_df = pd.DataFrame(embeddings, index=df.index).add_prefix('text_vec_')
    norm = Normalizer()
    norm_embeddings_df = pd.DataFrame(
        norm.fit_transform(embedding_df),
        index=df.index, columns=embedding_df.columns
    )

    # 3. Concatenate with Weights
    # We weight text embeddings (7.0) and card types (3.0) most heavily
    master_features = pd.concat([
        cmc_scaled * 1.5,
        pt_scaled * 0.5,
        keyword_dummies * 1.5,
        type_dummies * 3.0,
        norm_embeddings_df * 7.0 
    ], axis=1).fillna(0)
    
    # Ensure all columns are strings for PCA compatibility
    master_features.columns = master_features.columns.astype(str)
    return master_features

def run_pca(master_features, variance_threshold=0.95):
    """
    Reduces dimensionality using PCA while retaining specified variance.
    """
    pca = PCA(n_components=variance_threshold)
    pca_features = pca.fit_transform(master_features)
    
    df_pca = pd.DataFrame(
        pca_features, 
        index=master_features.index,
        columns=[f'PC{i+1}' for i in range(pca_features.shape[1])]
    )
    return df_pca

def find_similar_cards(card_name, df_metadata, pca_matrix, top_n=5):
    """
    Finds the most similar cards using Cosine Similarity in the PCA-reduced space.
    """
    try:
        query_idx = df_metadata[df_metadata['name'].str.lower() == card_name.lower()].index[0]
    except IndexError:
        return None

    query_vector = pca_matrix.iloc[query_idx].values.reshape(1, -1)
    scores = cosine_similarity(query_vector, pca_matrix).flatten()
    
    # Get indices of the highest scores (excluding the card itself)
    related_indices = scores.argsort()[-(top_n+1):][::-1]
    
    results = df_metadata.iloc[related_indices][['name', 'type_line', 'cmc', 'oracle_text']].copy()
    results['similarity_score'] = scores[related_indices]
    
    return results