##This file is crete embeddings which will remain fixed ###

### STEP 1: Load Co-occurrence Matrix from .RData ###
import pyreadr
import numpy as np
from scipy.sparse import dok_matrix

# Load the .RData file
result = pyreadr.read_r("Taxa_relative_abundance.RData")
first_key = list(result.keys())[0]
df = result[first_key]

# Extract taxa (vocabulary)
taxa = list(df.index)
print("Number of taxa:", len(taxa))

# Create vocab-index mappings
word2id = {taxon: idx for idx, taxon in enumerate(taxa)}
id2word = {idx: taxon for taxon, idx in word2id.items()}

# Create a sparse co-occurrence matrix
size = len(taxa)
cooc_matrix = dok_matrix((size, size), dtype=np.float32)
for i, row_taxon in enumerate(taxa):
    for j, col_taxon in enumerate(taxa):
        val = df.iat[i, j]
        if val > 0:
            cooc_matrix[i, j] = val

print("✅ Co-occurrence sparse matrix created.")


### STEP 2: Define GloVe Model ###
import torch
import torch.nn as nn

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max=100, alpha=0.75):
        super(GloVeModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha

        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, i_idx, j_idx, cooc_vals):
        wi = self.wi(i_idx)
        wj = self.wj(j_idx)
        bi = self.bi(i_idx).squeeze()
        bj = self.bj(j_idx).squeeze()

        dot = torch.sum(wi * wj, dim=1)
        log_x = torch.log(cooc_vals + 1e-10)

        weight = torch.pow(torch.clamp(cooc_vals / self.x_max, max=1.0), self.alpha)
        loss = weight * (dot + bi + bj - log_x) ** 2 #loss function
        return torch.sum(loss)


### STEP 3: Prepare Data for Training ###
from scipy.sparse import coo_matrix

cooc_matrix_coo = cooc_matrix.tocoo()
i_indices = torch.LongTensor(cooc_matrix_coo.row)
j_indices = torch.LongTensor(cooc_matrix_coo.col)
cooc_values = torch.FloatTensor(cooc_matrix_coo.data)

print("✅ Data ready for GloVe training.")


### STEP 4: Train the GloVe Model ###
embedding_dim = 128
model = GloVeModel(size, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05) #gradient descent 
epochs = 400

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model(i_indices, j_indices, cooc_values)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("✅ Training complete.")


### STEP 5: Extract and Save Embeddings to .CSV ###
import pandas as pd

with torch.no_grad():
    final_embeddings = model.wi.weight + model.wj.weight

embedding_array = final_embeddings.numpy()
taxa_embeddings = pd.DataFrame(embedding_array)
taxa_embeddings.insert(0, "Taxon", [id2word[i] for i in range(size)])

output_path = "glove_taxa_embeddings_128.csv"
taxa_embeddings.to_csv(output_path, index=False)

print(f"💾 Embeddings saved to: {output_path}")



