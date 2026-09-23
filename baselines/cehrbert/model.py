import math
import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        # tau: [B, L, in_features]
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], dim=-1)

class CehrBertEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_seq_len=300, time_dim=32, age_dim=32, dropout=0.1):
        super().__init__()
        self.concept_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embeddings = nn.Embedding(2, d_model)
        
        self.time_embeddings = Time2Vec(1, time_dim)
        self.age_embeddings = Time2Vec(1, age_dim)
        
        # In the CEHR-BERT paper, [e_concept; e_time; e_age; e_segment?] 
        # Wait, the paper concatenates [e_concept; e_time; e_age] and projects, then adds segment/position? 
        # Let's concatenate [concept, segment, time, age] and project to d_model, 
        # or concatenate [concept, time, age] -> linear -> add segment.
        # "Specifically, the concept embedding, visit-segment embedding, age embedding, and absolute-time embedding are concatenated and projected"
        self.fc = nn.Linear(d_model + d_model + time_dim + age_dim, d_model)
        
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids, time_stamps, ages):
        # time_stamps, ages: [B, L]
        # input_ids, segment_ids: [B, L]
        
        time_stamps = time_stamps.unsqueeze(-1) # [B, L, 1]
        ages = ages.unsqueeze(-1) # [B, L, 1]
        
        e_concept = self.concept_embeddings(input_ids)
        e_segment = self.segment_embeddings(segment_ids)
        e_time = self.time_embeddings(time_stamps)
        e_age = self.age_embeddings(ages)
        
        concat_emb = torch.cat([e_concept, e_segment, e_time, e_age], dim=-1)
        embeddings = self.fc(concat_emb)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class CehrBertPretrainModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=5, n_heads=8, dropout=0.1, max_seq_len=300):
        super().__init__()
        self.embeddings = CehrBertEmbeddings(vocab_size, d_model, max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, segment_ids, time_stamps, ages, attention_mask):
        # attention_mask: [B, L] where 1 is real, 0 is pad
        # nn.TransformerEncoder expects src_key_padding_mask where True means pad
        src_key_padding_mask = ~(attention_mask.bool())
        
        emb = self.embeddings(input_ids, segment_ids, time_stamps, ages)
        hidden_states = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        
        mlm_logits = self.mlm_head(hidden_states)
        return hidden_states, mlm_logits

class CehrBertPooledClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.classifier = nn.Linear(d_model, num_classes)
        # Note: final layer bias should probably be initialized to -7 for rare outcomes if used for T2D/etc,
        # but the prompt says: "Keep this head simple and use it for the main apples-to-apples representation comparison."

    def forward(self, input_ids, segment_ids, time_stamps, ages, attention_mask):
        # We don't need mlm_logits here, but we pass through the encoder
        src_key_padding_mask = ~(attention_mask.bool())
        emb = self.encoder.embeddings(input_ids, segment_ids, time_stamps, ages)
        hidden_states = self.encoder.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        
        # Take the [CLS] token representation if it exists, or max pool
        # Assuming the first token is [CLS] (or some aggregated representation)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.pooler(first_token_tensor)
        logits = self.classifier(pooled_output)
        return logits

class CehrBertBiLSTMClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes):
        super().__init__()
        self.encoder = encoder
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, segment_ids, time_stamps, ages, attention_mask):
        src_key_padding_mask = ~(attention_mask.bool())
        emb = self.encoder.embeddings(input_ids, segment_ids, time_stamps, ages)
        hidden_states = self.encoder.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        
        # Lengths for pack_padded_sequence
        lengths = attention_mask.sum(dim=1).cpu()
        # Ensure at least length 1 to avoid pack_padded errors
        lengths = torch.clamp(lengths, min=1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hn, cn) = self.bilstm(packed_input)
        
        # Use the final hidden state of both directions
        # hn is [2, B, hidden_size]
        hn = torch.cat([hn[0], hn[1]], dim=-1) # [B, d_model]
        logits = self.classifier(hn)
        return logits
