'''
@Author: Wang Dong
@Date: 2025.10.27
@Description: Data preprocessing and encodeing
@Negative sampling strategy: motif similarity based
'''

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

# -------------------------
# Rotation Attention
# -------------------------
def RotationBasedAttention(theta_a, theta_b, weight=None):
    cos_a, cos_b = torch.cos(theta_a), torch.cos(theta_b)
    sin_a, sin_b = torch.sin(theta_a), torch.sin(theta_b)
    if weight is not None:
        cos_a, sin_a = cos_a * weight, sin_a * weight
    return torch.sigmoid(cos_a @ cos_b.transpose(-2, -1) + sin_a @ sin_b.transpose(-2, -1))


class SelfAttentiveRotation(nn.Module):
    def __init__(self, config, input_shape, output_shape) -> None:
        super().__init__()
        self.input_shape, self.output_shape = input_shape, output_shape
        self.field_num = config.num_field
        self.head_num = config.head_num
        self.initialize_parameters(config.drop_rate_att)

    def initialize_parameters(self, dropout_prob):
        self.Q, self.K, self.V = [
            nn.Parameter(torch.ones(self.field_num, self.input_shape, self.output_shape))
            for _ in range(3)
        ]
        self.linear_transform = nn.Linear(self.input_shape, self.output_shape, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm([self.output_shape])
        self.weight = nn.Parameter(torch.ones(1, self.output_shape // self.head_num))
        for param in [self.Q, self.K, self.V, self.linear_transform.weight, self.weight]:
            xavier_normal_(param, gain=1.414)

    def forward(self, theta):
        tensor = theta.reshape(theta.shape[0], -1, self.input_shape)
        queries, keys, values = [torch.einsum('bfd,fdw->bfw', tensor, param) for param in [self.Q, self.K, self.V]]

        head_tensors = [
            torch.stack(torch.split(tensor, [self.output_shape // self.head_num] * self.head_num, dim=-1), dim=1)
            for tensor in [queries, keys, values]
        ]
        queries, keys, values = head_tensors

        scores = RotationBasedAttention(queries, keys, weight=self.weight).transpose(-2, -1)
        values = self.dropout(values).transpose(-2, -1)
        output = torch.cat(
            torch.split(
                torch.transpose(values @ scores, -2, -1),
                [1] * self.head_num, dim=1
            ), dim=-1
        ).squeeze(1)

        theta = self.dropout(theta)
        representation = self.norm(output + self.linear_transform(theta))
        return representation


class Amp(nn.Module):
    def __init__(self, config, input_shape, output_shape, activation=None) -> None:
        super().__init__()
        self.linear = nn.Linear(input_shape, output_shape)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        self.dropout = nn.Dropout(p=config.dp_rate_amp)
        self.activation = activation
        self.group_size = config.group_hidden_dimension
        self.norm1 = nn.LayerNorm([self.group_size])
        self.norm2 = nn.LayerNorm([self.group_size])

    def forward(self, kws):
        real, imag = kws
        real = self.dropout(real).view(real.size(0), -1)
        imag = self.dropout(imag).view(imag.size(0), -1)
        real, imag = self.linear(real), self.linear(imag)
        if self.activation:
            real, imag = self.activation(real), self.activation(imag)
        real = real.view(real.size(0), -1, self.group_size)
        imag = imag.view(imag.size(0), -1, self.group_size)
        real, imag = self.norm1(real), self.norm2(imag)
        return real, imag


class AmpNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        mlps = [Amp(config, in_shape, out_shape, nn.ReLU())
                for in_shape, out_shape in zip(config.mlp_list[:-1], config.mlp_list[1:])]
        self.block = nn.Sequential(*mlps)
        self.regularization = nn.Linear(config.mlp_list[-1], 1)
        nn.init.xavier_normal_(self.regularization.weight)

    def forward(self, feature):
        real, imag = feature
        real, imag = self.block((real, imag))
        real = real.view(real.size(0), -1)
        imag = imag.view(imag.size(0), -1)
        real, imag = self.regularization(real), self.regularization(imag)
        return real + imag


# -------------------------
# RFM Model
# -------------------------
class RFM(nn.Module):
    def __init__(self,
                 vocab_sizes,
                 embedding_size=64,
                 hidden_units=64,
                 att_layers=2,
                 head_num=4,
                 mlp_list=[128, 64],
                 group_hidden_dimension=8,
                 drop_rate_att=0.1,
                 dp_rate_amp=0.1,
                 add_first_order_residual=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.att_layers = att_layers
        self.head_num = head_num
        self.add_first_order_residual = add_first_order_residual
        self.dp_rate_amp = dp_rate_amp
        self.drop_rate_att = drop_rate_att
        self.group_hidden_dimension = group_hidden_dimension

        # categorical embedding
        semantic_fields = ["tsrna_type", "tsrna_type_1", "organ", "icd", "icd_1", "icd_2"]
        all_fields = semantic_fields

        self.embed_layers = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], embedding_size)
            for name in all_fields if name in vocab_sizes
        })
        for emb in self.embed_layers.values():
            nn.init.xavier_normal_(emb.weight)

        # PCA features
        self.tsrna_pca_dim = vocab_sizes.get("tsrna_pca_dim", 0)
        self.disease_pca_dim = vocab_sizes.get("disease_pca_dim", 0)
        self.motif_pca_dim = vocab_sizes.get("motif_pca_dim", 0)

        self.tsrna_pca_layers = nn.ModuleList(
            [nn.Linear(1, embedding_size) for _ in range(self.tsrna_pca_dim)]
        )
        self.disease_pca_layers = nn.ModuleList(
            [nn.Linear(1, embedding_size) for _ in range(self.disease_pca_dim)]
        )
        self.motif_pca_layers = nn.ModuleList(
            [nn.Linear(1, embedding_size) for _ in range(self.motif_pca_dim)]
        )
        for layer in list(self.tsrna_pca_layers) + list(self.disease_pca_layers):
            nn.init.xavier_normal_(layer.weight)
        for layer in self.motif_pca_layers:
            nn.init.xavier_normal_(layer.weight)

        # attention architecture
        self.attention_architecture = [embedding_size] + [hidden_units] * att_layers
        self.num_field = len(self.embed_layers) + \
                         self.tsrna_pca_dim + self.disease_pca_dim + self.motif_pca_dim
        self.saro_layers = nn.Sequential(
            *(SelfAttentiveRotation(self, in_shape, out_shape)
              for in_shape, out_shape in zip(self.attention_architecture[:-1], self.attention_architecture[1:]))
        )

        self.norm1 = nn.LayerNorm([self.attention_architecture[-1]])
        self.norm2 = nn.LayerNorm([self.attention_architecture[-1]])
        mlp_input_dim = self.attention_architecture[-1] * self.num_field
        self.ampnet = AmpNet(self._build_config(mlp_input_dim, mlp_list))

        self.projection = nn.Linear(embedding_size, self.attention_architecture[-1])
        nn.init.normal_(self.projection.weight, mean=0, std=0.01)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _build_config(self, mlp_input_dim, mlp_list):
        class Cfg: pass
        cfg = Cfg()
        cfg.mlp_list = [mlp_input_dim] + mlp_list
        cfg.group_hidden_dimension = self.group_hidden_dimension
        cfg.dp_rate_amp = self.dp_rate_amp
        return cfg

    def concat_embed_input_fields(self, batch):
        embed_list = [self.embed_layers[name](batch[name]) for name in self.embed_layers.keys()]
        cat_embed = torch.stack(embed_list, dim=1)  # (B, F_cat, D)

        # tsRNA PCA
        tsrna_pca_embeds = []
        for i, layer in enumerate(self.tsrna_pca_layers):
            col_name = f"tsrna_pca_{i}"
            if col_name in batch:
                x = batch[col_name].unsqueeze(1)  # (B, 1)
                tsrna_pca_embeds.append(layer(x))  # (B, D)
        if len(tsrna_pca_embeds) > 0:
            tsrna_pca_embeds = torch.stack(tsrna_pca_embeds, dim=1)  # (B, N_tsrna_pca, D)
        else:
            tsrna_pca_embeds = torch.empty(0, device=cat_embed.device)

        # disease PCA
        disease_pca_embeds = []
        for i, layer in enumerate(self.disease_pca_layers):
            col_name = f"disease_pca_{i}"
            if col_name in batch:
                x = batch[col_name].unsqueeze(1)
                disease_pca_embeds.append(layer(x))
        if len(disease_pca_embeds) > 0:
            disease_pca_embeds = torch.stack(disease_pca_embeds, dim=1)
        else:
            disease_pca_embeds = torch.empty(0, device=cat_embed.device)

        # motif PCA
        motif_pca_embeds = []
        for i, layer in enumerate(self.motif_pca_layers):
            col_name = f"motif_pca_{i}"
            if col_name in batch:
                x = batch[col_name].unsqueeze(1)  # (B, 1)
                motif_pca_embeds.append(layer(x))  # (B, D)
        if len(motif_pca_embeds) > 0:
            motif_pca_embeds = torch.stack(motif_pca_embeds, dim=1)  # (B, N_motif_pca, D)
        else:
            motif_pca_embeds = torch.empty(0, device=cat_embed.device)

        all_embeds = [cat_embed]
        if tsrna_pca_embeds.numel() > 0:
            all_embeds.append(tsrna_pca_embeds)
        if disease_pca_embeds.numel() > 0:
            all_embeds.append(disease_pca_embeds)
        if motif_pca_embeds.numel() > 0:
            all_embeds.append(motif_pca_embeds)

        return torch.cat(all_embeds, dim=1)  # (B, F_total, D)

    def forward(self, batch):
        angular_embeddings = self.concat_embed_input_fields(batch)
        theta = self.saro_layers(angular_embeddings)
        r, p = torch.cos(theta), torch.sin(theta)

        if self.add_first_order_residual:
            r = self.norm1(r + torch.cos(self.projection(angular_embeddings)))
            p = self.norm2(p + torch.sin(self.projection(angular_embeddings)))

        logits = self.ampnet((r, p)).squeeze(-1)
        return logits

    def calculate_loss(self, batch):
        labels = batch["label"].float()
        output = self.forward(batch)
        return self.loss_fn(output, labels)

    def predict(self, batch):
        return self.sigmoid(self.forward(batch))
