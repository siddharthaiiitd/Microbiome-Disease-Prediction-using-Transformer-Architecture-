# inference_app.py

import streamlit as st
import torch
import pandas as pd
import numpy as np
from dataprocess import MicrobiomeDataset
from model import MicrobiomeTransformerClassifier

# --- Load GloVe embeddings and setup vocab mappings ---
glove = pd.read_csv("glove_taxa_embeddings.csv", index_col=0)
taxa_names = list(glove.index)

seq_type_vocab = {'16s': 0, 'wgs': 1}
lifestyle_vocab = {
    'IndustrializedUrban': 0,
    'UrbanRuralMixed': 1,
    'RuralTribal': 2
}

# Positional Encoding function
def generate_positional_encoding(max_rank, dim):
    pe = np.zeros((max_rank, dim))
    position = np.arange(1, max_rank + 1).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe.astype(np.float32)

positional_encoding = generate_positional_encoding(425, glove.shape[1])

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MicrobiomeTransformerClassifier()
model.load_state_dict(torch.load("Final_microbiome_model_epoch45.pt", map_location=device))
model.to(device)
model.eval()

st.title("🧬 Microbiome Disease Prediction - Logit Viewer")

st.markdown("""
Upload a CSV with:
- **Row 1**: Column names (`taxa_1`, ..., `taxa_425`, `seq_type`, `cohort_life_style`)
- **Row 2**: 425 taxa abundance values + 2 metadata values
""")

uploaded_file = st.file_uploader("Upload CSV (2 rows)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.shape[0] != 1 or df.shape[1] != 427:
        st.error("CSV must have 1 data row and 427 columns (425 taxa + seq_type + cohort_life_style)")
    else:
        row = df.iloc[0]

        # Extract abundance
        abundance = row[:425].values.astype(np.float32)

        # Get rank
        rank_indices = np.argsort(-abundance)
        ranks = np.empty_like(rank_indices)
        ranks[rank_indices] = np.arange(1, 426)

        # Taxa embeddings
        taxa_embeddings = []
        mask = []

        for i, taxon in enumerate(taxa_names):
            emb = glove.loc[taxon].values.astype(np.float32)
            if abundance[i] > 0:
                pe = positional_encoding[ranks[i] - 1]
                final_emb = emb + pe
                mask.append(False)
            else:
                final_emb = np.zeros_like(emb)
                mask.append(True)
            taxa_embeddings.append(final_emb)

        taxa_embeddings = np.stack(taxa_embeddings)  # [425, 100]
        mask = np.array(mask)                        # [425]

        # Metadata
        try:
            seq_type = row['seq_type'].strip().lower()
            lifestyle = row['cohort_life_style'].strip()
            seq_emb = torch.nn.Embedding(2, glove.shape[1])(torch.tensor(seq_type_vocab[seq_type])).detach().numpy()
            life_emb = torch.nn.Embedding(3, glove.shape[1])(torch.tensor(lifestyle_vocab[lifestyle])).detach().numpy()
        except KeyError as e:
            st.error(f"Invalid metadata value: {e}")
            st.stop()

        # Combine
        input_tensor = np.vstack([taxa_embeddings, seq_emb, life_emb])  # [427, 100]
        mask = np.concatenate([mask, [False, False]])  # [427]

        # Convert to torch
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 427, 100]
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)              # [1, 427]

        # Inference
        import torch.nn.functional as F

        # Inference
        with torch.no_grad():
            logits = model(input_tensor, mask_tensor)  # [1, 2]
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

        st.subheader("Prediction Probabilities:")
        st.write(f"🟢 **Control**: {probs[0] * 100:.2f}%")
        st.write(f"🔴 **Disease**: {probs[1] * 100:.2f}%")
