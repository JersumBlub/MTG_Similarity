# MTG Semantic Similarity Engine
**A high-performance search tool for Magic: The Gathering that identifies mechanically similar cards using semantic embeddings and PCA.**


## Project Overview
Standard card databases often rely on keyword matching, making it difficult to find cards that are functionally identical but worded differently. This engine uses **Natural Language Processing (NLP)** and **Dimensionality Reduction** to build a semantic map of the MTG multiverse.

Instead of just finding cards with the word "Fireball," this engine finds cards that *act* like a Fireball—prioritizing mechanical behavior, power level, and card role.

## Key Technical Features
* **Entity Masking:** Includes a custom preprocessing step that replaces a card's name in its own text with `[SELF]` (e.g., *Questing Beast* becomes *[SELF]*). This prevents the model from over-biasing similarity based on card names.
* **Transformer-Based Semantic Search:** Uses the `all-MiniLM-L6-v2` Sentence-Transformer model to capture the deep context of card abilities, which is more effective for "keyword-heavy" games than traditional TF-IDF.
* **Weighted Multi-Modal Input:** Consolidates diverse data types into a single master matrix with custom weighting:
    * **Text Semantics (7.0x):** The primary driver of mechanical identity.
    * **Card Types (3.0x):** Heavy weight to ensure Creatures are matched with Creatures.
    * **Keywords & Stats (1.5x):** Nuanced weighting for mana cost and specific mechanics.
* **Efficient Vector Space:** Employs **Principal Component Analysis (PCA)** to retain **95% of the variance** in a dense representation, making high-speed similarity calculations possible without storing a massive 30k x 30k matrix.

## Repository Structure
```text
mtg-similarity/
├── data/               # Project data (scryfall_oracle.csv)
├── src/                # Modular logic for production
│   ├── __init__.py
│   ├── processing.py   # Regex cleaning, stat parsing, and masking
│   └── model.py        # Embedding generation, PCA, and similarity logic
├── main.py             # Entry point to run the end-to-end pipeline
└── requirements.txt    # Project dependencies
```

## Getting Started

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JersumBlub/mtg-similarity.git
    cd mtg-similarity
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Acquisition

To keep the repository lightweight and comply with GitHub's file size limits, the raw card data is not included in this repository. 

Follow these steps to set up the data:

1.  **Download the Data**: Visit [Scryfall's Bulk Data Page](https://scryfall.com/docs/api/bulk-data).
2.  **Select the File**: Locate the **Oracle Cards** section and download the **Default Cards** file in **CSV** format.
3.  **Local Setup**:
    * Create a folder named `data/` in the project root.
    * Move the downloaded file into this folder.
    * Ensure the file is named `scryfall_oracle.csv` (or update the filename in `main.py`).

> **Note:** The `.gitignore` file is configured to prevent this CSV from being uploaded to GitHub, ensuring the repository remains under 100MB.

### Running the Engine
Ensure your `scryfall_oracle.csv` is in the `data/` directory. Then, execute the main script to process the data and run test queries:
```bash
python main.py
```

## Sample Results
The engine identifies mechanical siblings by analyzing weighted features including card types, keywords, and semantic oracle text. Based on the similarity search logic implemented in the project, here are some example outputs:

| Query Card | Top Semantic Matches | Similarity Score |
| :--- | :--- | :--- |
| **Giada, Font of Hope** | Youthful Valkyrie, Herald of War, Skyline Savior | 0.72 - 0.58 |
| **Llanowar Elves** | Fyndhorn Elves, Elvish Mystic, Arbor Elf | 0.85 - 0.98 |
| **Lightning Bolt** | Shock, Galvanic Blast, Chain Lightning | 0.91 - 0.95 |

---

## Technologies Used
* **Python 3.10+**: The core language for data processing and model orchestration.
* **Pandas & NumPy**: Used for robust data manipulation, stat parsing, and matrix operations.
* **Scikit-Learn**: Utilized for `MinMaxScaler`, `MultiLabelBinarizer`, and `PCA` (configured for 95% variance retention).
* **Sentence-Transformers**: Specifically the `all-MiniLM-L6-v2` model for generating high-performance semantic text embeddings.