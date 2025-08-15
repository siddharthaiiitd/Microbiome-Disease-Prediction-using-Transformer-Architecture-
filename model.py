import torch
import torch.nn as nn

class MicrobiomeTransformerClassifier(nn.Module):
    def __init__(self, embedding_dim=100, num_layers=6, num_heads=4, ff_dim=256, dropout=0.1):
        super(MicrobiomeTransformerClassifier, self).__init__()

        self.embedding_dim = embedding_dim

        # Learnable CLS token, shape [1, 100]; only one copy needed for parameter sharing
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))  # ← [1, 100]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # ensures input shape is [B, T, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head on top of CLS
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # binary classification
        )

    def forward(self, x, mask):
        """
        Args:
            x: [B, 427, 100]  Input tensor without CLS token
            mask: [B, 427]    Boolean mask (True = ignore)

        Returns:
            logits: [B, 2]
        """
        B = x.size(0)

        # Repeat CLS token across batch: [B, 1, 100] — separate copies for each sample
        cls_token_batch = self.cls_token.unsqueeze(0).repeat(B, 1, 1)  # ← important fix

        # Prepend CLS token to input sequence → [B, 428, 100]
        x = torch.cat([cls_token_batch, x], dim=1)

        # Update mask to include CLS token (always unmasked) → [B, 428]
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Pass through Transformer encoder with mask
        encoded = self.encoder(x, src_key_padding_mask=mask)  # [B, 428, 100]

        # Extract updated CLS token (position 0) → [B, 100]
        cls_repr = encoded[:, 0, :]

        # MLP classifier → [B, 2]
        logits = self.classifier(cls_repr)

        return logits
