import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

class MicrobiomeDataset(Dataset):
    def __init__(self, data_path, glove_path, has_labels=True):  # === CHANGED: added has_labels flag ===
        """
        Args:
            data_path (str): Path to samples.csv
            glove_path (str): Path to glove.csv (precomputed taxa embeddings)
            has_labels (bool): Whether the dataset contains 'diseaseCat' labels (True during training, False during testing)
        """
        # Load sample data and embeddings using passed paths
        self.data = pd.read_csv(data_path)
        self.glove = pd.read_csv(glove_path, index_col=0)

        # Save label availability flag
        self.has_labels = has_labels  # === CHANGED ===

        # Remaining setup
        self.taxa_names = list(self.glove.index)
        self.embedding_dim = self.glove.shape[1]

        self.seq_type_vocab = {'16s': 0, 'wgs': 1}
        
        # === CHANGED: Updated lifestyle_vocab to match updated values ===
        self.lifestyle_vocab = {
            'Industrialized': 0,
            'Non-Industrialized': 1
        }

        self.positional_encoding = self._generate_positional_encoding(max_rank=477, dim=self.embedding_dim)

        # Fixed random embeddings
        np.random.seed(42)
        self.seq_type_embedding = {
            0: np.random.normal(scale=0.1, size=self.embedding_dim).astype(np.float32),
            1: np.random.normal(scale=0.1, size=self.embedding_dim).astype(np.float32)
        }

        # === CHANGED: Updated lifestyle_embedding to match new vocab size ===
        self.lifestyle_embedding = {
            0: np.random.normal(scale=0.1, size=self.embedding_dim).astype(np.float32),
            1: np.random.normal(scale=0.1, size=self.embedding_dim).astype(np.float32)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a row from the dataset
        row = self.data.iloc[idx]
        # Generate input tensor, mask, and label
        input_tensor, mask_tensor, label_tensor = self.create_sample_tensors(row)
        return {
            'input': input_tensor,     # [479, 100]
            'mask': mask_tensor,       # [479]
            'label': label_tensor      # [1] or dummy if has_labels=False
        }

    def create_sample_tensors(self, row):
        """
        Create input tensor, mask tensor, and label from a single row of data.
        Args:
            row (pd.Series): A row from samples.csv

        Returns:
            input_tensor: torch.FloatTensor of shape [479, 100]
            mask_tensor: torch.BoolTensor of shape [479]
            label_tensor: torch.LongTensor scalar (0 or 1), or -1 if not available
        """
        # Get abundance vector
        abundance = row[self.taxa_names].values.astype(np.float32)  # [477]

        # Get rank of each taxon
        rank_indices = np.argsort(-abundance)
        ranks = np.empty_like(rank_indices)
        ranks[rank_indices] = np.arange(1, 478)

        # Build taxa embeddings with PE
        taxa_embeddings = []
        mask = []

        for i, taxon in enumerate(self.taxa_names):
            emb = self.glove.loc[taxon].values.astype(np.float32)
            if abundance[i] > 0:
                pe = self.positional_encoding[ranks[i] - 1]
                final_emb = emb + pe
                mask.append(False)
            else:
                final_emb = np.zeros_like(emb)
                mask.append(True)
            taxa_embeddings.append(final_emb)

        taxa_embeddings = np.stack(taxa_embeddings)  # [477, 100]
        mask = np.array(mask)                        # [477]

        # Embed metadata
        seq_type_id = self.seq_type_vocab[row['seq_type'].lower()]
        lifestyle_id = self.lifestyle_vocab[row['cohort_life_style']]  # Assumes preprocessed CSV matches updated vocab
        seq_emb = self.seq_type_embedding[seq_type_id]
        life_emb = self.lifestyle_embedding[lifestyle_id]

        # Combine embeddings
        input_tensor = np.vstack([taxa_embeddings, seq_emb, life_emb])  # [479, 100]
        mask = np.concatenate([mask, [False, False]])  # [479]

        # Label
        if self.has_labels:  # === CHANGED: conditional label creation ===
            label = 0 if row['diseaseCat'] == 'Control' else 1
        else:
            label = -1  # dummy label during inference

        return (
            torch.tensor(input_tensor, dtype=torch.float32),     # [479, 100]
            torch.tensor(mask, dtype=torch.bool),                # [479]
            torch.tensor(label, dtype=torch.long)                # [1]
        )

    def _generate_positional_encoding(self, max_rank, dim):
        """
        Generate sinusoidal positional encodings for taxa rank.
        """
        pe = np.zeros((max_rank, dim))
        position = np.arange(1, max_rank + 1).reshape(-1, 1)
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe.astype(np.float32)
