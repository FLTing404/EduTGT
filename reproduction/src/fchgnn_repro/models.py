from __future__ import annotations

import math

import torch
from torch import nn


def _joint_features(hidden: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
    width = min(hidden.shape[-1], target_feature.shape[-1])
    interaction = hidden[..., :width] * target_feature[..., :width]
    return torch.cat([hidden, target_feature, interaction], dim=-1)


class NextItemHead(nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim + min(hidden_dim, feature_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
        return self.net(_joint_features(hidden, target_feature)).squeeze(-1)


class EvidentialBinaryHead(nn.Module):
    """Beta-evidential Bernoulli head with explicit predictive uncertainty."""

    def __init__(self, hidden_dim: int, feature_dim: int, dropout: float):
        super().__init__()
        input_dim = hidden_dim + feature_dim + min(hidden_dim, feature_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hidden: torch.Tensor, target_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evidence = torch.nn.functional.softplus(self.net(_joint_features(hidden, target_feature)))
        alpha, beta = evidence.unbind(dim=-1)
        return alpha + 1.0, beta + 1.0


class GuidedResponseDiffusion(nn.Module):
    """Small conditional diffusion plug-in for binary response generation."""

    def __init__(self, context_dim: int, hidden_dim: int, steps: int):
        super().__init__()
        self.steps = int(steps)
        self.context = nn.Linear(context_dim, hidden_dim)
        self.time = nn.Embedding(self.steps, hidden_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        betas = torch.linspace(1e-4, 0.02, self.steps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def predict_noise(self, context: torch.Tensor, noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        encoded = self.context(context)
        time = self.time(timestep)
        return self.denoiser(torch.cat([encoded, time, noisy.unsqueeze(-1)], dim=-1)).squeeze(-1)

    def training_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected_context = context[mask]
        x0 = target[mask] * 2.0 - 1.0
        timestep = torch.randint(self.steps, (len(x0),), device=x0.device)
        alpha_bar = self.alpha_bars[timestep]
        noise = torch.randn_like(x0)
        noisy = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        predicted = self.predict_noise(selected_context, noisy, timestep)
        return torch.nn.functional.mse_loss(predicted, noise)

    def generate_logit(self, context: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(context.shape[:-1], device=context.device, dtype=context.dtype)
        for step in range(self.steps - 1, -1, -1):
            timestep = torch.full(x.shape, step, device=x.device, dtype=torch.long)
            alpha_bar = self.alpha_bars[step]
            noise = self.predict_noise(context, x, timestep)
            x0 = ((x - (1.0 - alpha_bar).sqrt() * noise) / alpha_bar.sqrt()).clamp(-1.0, 1.0)
            if step:
                previous = self.alpha_bars[step - 1]
                x = previous.sqrt() * x0 + (1.0 - previous).sqrt() * noise
            else:
                x = x0
        probability = ((x + 1.0) / 2.0).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(probability)


class DKT(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, embedding_dim: int, hidden_dim: int, dropout: float, **_: object):
        super().__init__()
        self.item = nn.Embedding(num_items, embedding_dim)
        self.concept = nn.Embedding(num_concepts, embedding_dim)
        self.response = nn.Embedding(2, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim * 3, hidden_dim, batch_first=True)
        self.head = NextItemHead(hidden_dim, embedding_dim * 2, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            self.item(batch["item"]),
            self.concept(batch["concept"]),
            self.response(batch["correct"].long()),
        ], dim=-1)
        hidden, _ = self.rnn(x)
        target = torch.cat([self.item(batch["next_item"]), self.concept(batch["next_concept"])], dim=-1)
        return self.head(hidden, target)


class SAINTStyle(nn.Module):
    def __init__(self, num_items: int, num_concepts: int, embedding_dim: int, hidden_dim: int, dropout: float, num_heads: int, max_length: int, **_: object):
        super().__init__()
        self.item = nn.Embedding(num_items, embedding_dim)
        self.concept = nn.Embedding(num_concepts, embedding_dim)
        self.response = nn.Embedding(2, embedding_dim)
        self.position = nn.Embedding(max_length, embedding_dim)
        self.input = nn.Linear(embedding_dim * 3, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # MPS does not implement the optional nested-tensor mask operator.
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.target = nn.Linear(embedding_dim * 2, hidden_dim)
        self.head = NextItemHead(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, length = batch["item"].shape
        pos = torch.arange(length, device=batch["item"].device).unsqueeze(0).expand(batch_size, -1)
        x = torch.cat([
            self.item(batch["item"]) + self.position(pos),
            self.concept(batch["concept"]),
            self.response(batch["correct"].long()),
        ], dim=-1)
        x = self.input(x)
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(x, mask=causal, src_key_padding_mask=~batch["mask"])
        target = self.target(torch.cat([self.item(batch["next_item"]), self.concept(batch["next_concept"])], dim=-1))
        return self.head(hidden, target)


class HypergraphLayer(nn.Module):
    def __init__(self, dim: int, dropout: float, attentive: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.attentive = attentive
        if attentive:
            self.node_gate = nn.Linear(dim, dim, bias=False)
            self.concept_gate = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        nodes: torch.Tensor,
        concepts: torch.Tensor,
        edge_item: torch.Tensor,
        edge_concept: torch.Tensor,
        num_concepts: int,
    ) -> torch.Tensor:
        transformed = self.linear(nodes)
        incident_nodes = transformed[edge_item]
        if self.attentive:
            scores = (
                self.node_gate(incident_nodes) * self.concept_gate(concepts[edge_concept])
            ).sum(dim=-1) / math.sqrt(transformed.shape[-1])
            node_to_edge_weight = torch.sigmoid(scores).unsqueeze(-1)
        else:
            node_to_edge_weight = torch.ones(
                len(edge_item), 1, device=nodes.device, dtype=nodes.dtype
            )
        hyper = torch.zeros(num_concepts, transformed.shape[-1], device=nodes.device, dtype=nodes.dtype)
        hyper.index_add_(0, edge_concept, incident_nodes * node_to_edge_weight)
        hyper_degree = torch.zeros(num_concepts, 1, device=nodes.device, dtype=nodes.dtype)
        hyper_degree.index_add_(0, edge_concept, node_to_edge_weight)
        hyper_degree.clamp_min_(1e-6)
        hyper = hyper / hyper_degree
        if self.attentive:
            edge_to_node_weight = torch.sigmoid(
                (incident_nodes * hyper[edge_concept]).sum(dim=-1) / math.sqrt(transformed.shape[-1])
            ).unsqueeze(-1)
        else:
            edge_to_node_weight = node_to_edge_weight
        out = torch.zeros_like(transformed)
        out.index_add_(0, edge_item, hyper[edge_concept] * edge_to_node_weight)
        node_degree = torch.zeros(nodes.shape[0], 1, device=nodes.device, dtype=nodes.dtype)
        node_degree.index_add_(0, edge_item, edge_to_node_weight)
        node_degree.clamp_min_(1e-6)
        out = out / node_degree
        return self.norm(nodes + self.dropout(torch.relu(out)))


class FCHGNN(nn.Module):
    """Reasonable reconstruction: multiscale item–concept hypergraph + FM cross + LSTM."""

    def __init__(
        self,
        num_items: int,
        num_concepts: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        hypergraph_layers: int,
        edge_item: torch.Tensor,
        edge_concept: torch.Tensor,
        difficulty: torch.Tensor,
        time_aware: bool = False,
        multi_concept: bool = False,
        hypergraph_attention: bool = False,
        dynamic_scale_fusion: bool = False,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.2,
        uncertainty_enabled: bool = False,
        uncertainty_kl_weight: float = 0.001,
        guided_diffusion: bool = False,
        diffusion_steps: int = 8,
        diffusion_loss_weight: float = 0.05,
        diffusion_logit_weight: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.time_aware = bool(time_aware)
        self.multi_concept = bool(multi_concept)
        self.dynamic_scale_fusion = bool(dynamic_scale_fusion)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_kl_weight = float(uncertainty_kl_weight)
        self.guided_diffusion = bool(guided_diffusion)
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.diffusion_logit_weight = float(diffusion_logit_weight)
        self.item = nn.Embedding(num_items, embedding_dim)
        self.concept = nn.Embedding(num_concepts, embedding_dim)
        self.response = nn.Embedding(2, embedding_dim)
        self.difficulty_projection = nn.Linear(1, embedding_dim)
        self.time_projection = nn.Sequential(nn.Linear(1, embedding_dim), nn.Tanh())
        self.layers = nn.ModuleList(
            [HypergraphLayer(embedding_dim, dropout, hypergraph_attention) for _ in range(hypergraph_layers)]
        )
        self.scale_logits = nn.Parameter(torch.zeros(hypergraph_layers + 1))
        self.scale_gate = nn.Linear(embedding_dim, hypergraph_layers + 1)
        field_count = 4 if self.time_aware else 3
        self.input_projection = nn.Linear(embedding_dim * field_count, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True)
        self.head = NextItemHead(hidden_dim, embedding_dim * 2, dropout)
        self.evidential_head = EvidentialBinaryHead(hidden_dim, embedding_dim * 2, dropout)
        joint_dim = hidden_dim + embedding_dim * 2 + min(hidden_dim, embedding_dim * 2)
        self.response_diffusion = GuidedResponseDiffusion(joint_dim, hidden_dim, diffusion_steps)
        self.contrastive_item = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.contrastive_concept = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.register_buffer("edge_item", edge_item.long())
        self.register_buffer("edge_concept", edge_concept.long())
        self.register_buffer("difficulty", difficulty.float())
        membership = torch.zeros(num_items, num_concepts, dtype=torch.bool)
        membership[edge_item.long(), edge_concept.long()] = True
        self.register_buffer("item_concept_membership", membership)

    def graph_items(self) -> torch.Tensor:
        scales = [self.item.weight]
        state = self.item.weight
        for layer in self.layers:
            state = layer(state, self.concept.weight, self.edge_item, self.edge_concept, self.num_concepts)
            scales.append(state)
        stacked = torch.stack(scales, dim=1)
        if self.dynamic_scale_fusion:
            weights = torch.softmax(self.scale_gate(self.item.weight), dim=-1).unsqueeze(-1)
        else:
            weights = torch.softmax(self.scale_logits, dim=0).view(1, -1, 1)
        return (stacked * weights).sum(dim=1)

    @staticmethod
    def fm_cross(fields: torch.Tensor) -> torch.Tensor:
        return 0.5 * (fields.sum(dim=-2).pow(2) - fields.pow(2).sum(dim=-2))

    def all_concept_features(self, item_graph: torch.Tensor) -> torch.Tensor:
        incident_items = item_graph[self.edge_item]
        incident_concepts = self.concept(self.edge_concept)
        scores = (incident_items * incident_concepts).sum(dim=-1) / math.sqrt(item_graph.shape[-1])
        weights = torch.sigmoid(scores).unsqueeze(-1)
        pooled = torch.zeros_like(item_graph)
        pooled.index_add_(0, self.edge_item, incident_concepts * weights)
        normalizer = torch.zeros(item_graph.shape[0], 1, device=item_graph.device, dtype=item_graph.dtype)
        normalizer.index_add_(0, self.edge_item, weights)
        return pooled / normalizer.clamp_min(1e-6)

    def feature(
        self,
        item_idx: torch.Tensor,
        concept_idx: torch.Tensor,
        item_graph: torch.Tensor,
        time_gap: torch.Tensor | None = None,
        all_concepts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item = item_graph[item_idx]
        concept = all_concepts[item_idx] if self.multi_concept and all_concepts is not None else self.concept(concept_idx)
        difficulty = self.difficulty_projection(self.difficulty[item_idx].unsqueeze(-1))
        fields = [item, concept, difficulty]
        if self.time_aware:
            if time_gap is None:
                time_gap = torch.zeros_like(item_idx, dtype=item.dtype)
            fields.append(self.time_projection(time_gap.unsqueeze(-1)))
        stacked = torch.stack(fields, dim=-2)
        linear = self.input_projection(torch.cat(fields, dim=-1))
        return torch.cat([linear, self.fm_cross(stacked)], dim=-1)

    def representations(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_graph = self.graph_items()
        all_concepts = self.all_concept_features(item_graph) if self.multi_concept else None
        current = self.feature(
            batch["item"], batch["concept"], item_graph, batch.get("time_gap"), all_concepts
        )
        response = self.response(batch["correct"].long())
        current = torch.cat(
            [current[..., : self.item.embedding_dim], current[..., self.item.embedding_dim :] + response], dim=-1
        )
        hidden, _ = self.rnn(current)
        target = self.feature(
            batch["next_item"], batch["next_concept"], item_graph, batch.get("next_time_gap"), all_concepts
        )
        return hidden, target, item_graph

    def concept_contrastive_loss(self, item_graph: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        used_items = torch.cat([batch["item"][batch["mask"]], batch["next_item"][batch["mask"]]])
        unique_items = torch.unique(used_items)
        item_z = torch.nn.functional.normalize(self.contrastive_item(item_graph[unique_items]), dim=-1)
        concept_z = torch.nn.functional.normalize(self.contrastive_concept(self.concept.weight), dim=-1)
        logits = item_z @ concept_z.transpose(0, 1) / max(self.contrastive_temperature, 1e-4)
        positives = self.item_concept_membership[unique_items].to(logits.dtype)
        log_prob = torch.log_softmax(logits, dim=-1)
        return -((log_prob * positives).sum(dim=-1) / positives.sum(dim=-1).clamp_min(1.0)).mean()

    @staticmethod
    def evidential_loss(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        kl_weight: float,
    ) -> torch.Tensor:
        alpha, beta, target = alpha[mask], beta[mask], target[mask]
        total = alpha + beta
        nll = target * (torch.digamma(total) - torch.digamma(alpha))
        nll += (1.0 - target) * (torch.digamma(total) - torch.digamma(beta))
        adjusted_alpha = target + (1.0 - target) * alpha
        adjusted_beta = (1.0 - target) + target * beta
        adjusted_total = adjusted_alpha + adjusted_beta
        kl = torch.lgamma(adjusted_total) - torch.lgamma(adjusted_alpha) - torch.lgamma(adjusted_beta)
        kl += (adjusted_alpha - 1.0) * (torch.digamma(adjusted_alpha) - torch.digamma(adjusted_total))
        kl += (adjusted_beta - 1.0) * (torch.digamma(adjusted_beta) - torch.digamma(adjusted_total))
        return nll.mean() + kl_weight * kl.mean()

    def training_objective(
        self,
        batch: dict[str, torch.Tensor],
        epoch: int,
        total_epochs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, target_feature, item_graph = self.representations(batch)
        if self.uncertainty_enabled:
            alpha, beta = self.evidential_head(hidden, target_feature)
            logits = torch.log(alpha) - torch.log(beta)
            annealed_kl = self.uncertainty_kl_weight * min(1.0, epoch / max(total_epochs // 2, 1))
            loss = self.evidential_loss(alpha, beta, batch["target"], batch["mask"], annealed_kl)
        else:
            logits = self.head(hidden, target_feature)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[batch["mask"]], batch["target"][batch["mask"]]
            )
        if self.contrastive_weight > 0.0:
            loss = loss + self.contrastive_weight * self.concept_contrastive_loss(item_graph, batch)
        if self.guided_diffusion and self.diffusion_loss_weight > 0.0:
            context = _joint_features(hidden, target_feature)
            diffusion_loss = self.response_diffusion.training_loss(context, batch["target"], batch["mask"])
            loss = loss + self.diffusion_loss_weight * diffusion_loss
        return loss, logits

    def prediction_details(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden, target_feature, _ = self.representations(batch)
        uncertainty = None
        if self.uncertainty_enabled:
            alpha, beta = self.evidential_head(hidden, target_feature)
            logits = torch.log(alpha) - torch.log(beta)
            uncertainty = 2.0 / (alpha + beta)
        else:
            logits = self.head(hidden, target_feature)
        if self.guided_diffusion and self.diffusion_logit_weight > 0.0:
            context = _joint_features(hidden, target_feature)
            logits = logits + self.diffusion_logit_weight * self.response_diffusion.generate_logit(context)
        return logits, uncertainty

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.prediction_details(batch)[0]

    def predictive_uncertainty(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        return self.prediction_details(batch)[1]


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name in {"akt", "simplekt"}:
        from .modern_baselines import AKTOfficial, SimpleKTOfficial

        return {"akt": AKTOfficial, "simplekt": SimpleKTOfficial}[name](**kwargs)
    if name == "dkt":
        return DKT(**kwargs)
    if name == "saint":
        return SAINTStyle(**kwargs)
    if name == "fchgnn":
        return FCHGNN(**kwargs)
    raise ValueError(f"Unknown neural model: {name}")
