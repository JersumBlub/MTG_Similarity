import pandas as pd
from src.processing import preprocess_dataframe, encode_categorical_features
from src.model import (
    generate_embeddings, 
    build_feature_matrix, 
    run_pca, 
    find_similar_cards
)

def main():
    # 1. Load Data
    print("Step 1: Loading Scryfall data...")
    df_raw = pd.read_csv('data/scryfall_oracle_20260102.csv', low_memory=False)
    
    # 2. Preprocess (Cleaning & Masking)
    print("Step 2: Cleaning and masking names...")
    df = preprocess_dataframe(df_raw)
    
    # 3. Encoding (One-hot Keywords & Types)
    print("Step 3: Encoding categorical features...")
    type_dummies, keyword_dummies = encode_categorical_features(df)
    
    # 4. Embeddings (Semantic Text)
    print("Step 4: Generating semantic text embeddings...")
    # Using only a subset or masked_text column as defined in your notebook
    embeddings = generate_embeddings(df['masked_text'].fillna('').tolist())
    
    # 5. Build Master Feature Set
    print("Step 5: Building weighted feature matrix...")
    master_features = build_feature_matrix(df, type_dummies, keyword_dummies, embeddings)
    
    # 6. Dimensionality Reduction
    print("Step 6: Running PCA for 95% variance...")
    df_pca = run_pca(master_features)
    
    # 7. Test the Similarity Engine
    test_card = "Llanowar Elves"
    print(f"\nFinding cards similar to: {test_card}")
    results = find_similar_cards(test_card, df, df_pca)
    
    if results is not None:
        print(results[['name', 'similarity_score']])
    else:
        print("Card not found in dataset.")

if __name__ == "__main__":
    main()