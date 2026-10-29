"""Official pyKT AKT and simpleKT adapted to the project's next-item batches.

The neural architectures are vendored, not reimplemented. See vendor/pykt for
the pinned source, MIT license, checksums, and minimal device/logit patches.
"""

from __future__ import annotations

import torch
from torch import nn

from .vendor.pykt.akt import AKT
from .vendor.pykt.simplekt import simpleKT


def _full_sequences(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build q[0:L+1], c[0:L+1], r[0:L+1] without reading target labels.

    r[t] is only available to predictions after t in the official attention
    mask. The final unknown response has a placeholder that is never attended.
    Reserve ID 0 for trailing padding; project item/concept IDs are zero based.
    """
    # Loss masks may exclude real events (e.g. duplicate KC targets). The loader
    # keeps event IDs for those events, so preserve them as observed history.
    if "event_idx" in batch:
        observed = batch["event_idx"] >= 0
    else:
        # Small/custom batches without event IDs must be right-padded. Include
        # every history position up to the last supervised target.
        positions = torch.arange(1, batch["mask"].shape[1] + 1, device=batch["mask"].device)
        last = (batch["mask"].long() * positions).max(dim=1, keepdim=True).values
        observed = positions <= last
    valid = torch.cat([observed[:, :1], observed], dim=1)
    items = torch.cat([batch["item"][:, :1], batch["next_item"]], dim=1) + 1
    concepts = torch.cat([batch["concept"][:, :1], batch["next_concept"]], dim=1) + 1
    responses = torch.cat([batch["correct"].long(), torch.zeros_like(batch["correct"][:, :1], dtype=torch.long)], dim=1)
    return items.masked_fill(~valid, 0), concepts.masked_fill(~valid, 0), responses.masked_fill(~valid, 0)


class AKTOfficial(nn.Module):
    """pyKT's AKT-Rasch (qid) with its original monotonic attention and L2.

    embedding_dim maps to d_model, hidden_dim to d_ff/final_fc_dim. AKT's
    official second prediction layer remains 256 wide. These are controlled
    project hyperparameters, not the published paper's dataset-specific tuning.
    """

    def __init__(
        self, num_items: int, num_concepts: int, embedding_dim: int = 64,
        hidden_dim: int = 128, dropout: float = 0.2, num_heads: int = 4,
        max_length: int = 100, n_blocks: int = 1, d_ff: int | None = None,
        final_fc_dim: int | None = None, kq_same: int = 1,
        separate_qa: bool = False, l2: float = 1e-5, **_: object,
    ):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.max_length = int(max_length)
        self.core = AKT(
            n_question=num_concepts + 1, n_pid=num_items, d_model=embedding_dim,
            n_blocks=n_blocks, dropout=dropout, d_ff=d_ff or hidden_dim,
            kq_same=kq_same, final_fc_dim=final_fc_dim or hidden_dim,
            num_attn_heads=num_heads, separate_qa=separate_qa, l2=l2, emb_type="qid",
        )

    def _predict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if batch["item"].shape[1] > self.max_length:
            raise ValueError("batch length exceeds configured max_length")
        items, concepts, responses = _full_sequences(batch)
        logits, regularizer = self.core(concepts, responses, items, return_logits=True)
        return logits[:, 1:], regularizer

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._predict(batch)[0]

    def training_objective(
        self, batch: dict[str, torch.Tensor], epoch: int = 1, total_epochs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, regularizer = self._predict(batch)
        mask = batch["mask"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], batch["target"][mask])
        return loss + regularizer, logits


class SimpleKTOfficial(nn.Module):
    """Author-official simpleKT with vector question difficulty (qid).

    Its optional scalar/no-Rasch variants are deliberately not substituted.
    max_length counts next-item predictions; the official full sequence needs
    one additional positional embedding for the initial observed interaction.
    """

    def __init__(
        self, num_items: int, num_concepts: int, embedding_dim: int = 64,
        hidden_dim: int = 128, dropout: float = 0.2, num_heads: int = 4,
        max_length: int = 100, n_blocks: int = 1, d_ff: int | None = None,
        final_fc_dim: int | None = None, final_fc_dim2: int | None = None,
        kq_same: int = 1, separate_qa: bool = False, **_: object,
    ):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if embedding_dim % 2:
            raise ValueError("simpleKT sinusoidal positions require even embedding_dim")
        self.max_length = int(max_length)
        self.core = simpleKT(
            n_question=num_concepts + 1, n_pid=num_items, d_model=embedding_dim,
            n_blocks=n_blocks, dropout=dropout, d_ff=d_ff or hidden_dim,
            kq_same=kq_same, final_fc_dim=final_fc_dim or hidden_dim,
            final_fc_dim2=final_fc_dim2 or hidden_dim, num_attn_heads=num_heads,
            separate_qa=separate_qa, seq_len=self.max_length + 1, emb_type="qid",
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if batch["item"].shape[1] > self.max_length:
            raise ValueError("batch length exceeds configured max_length")
        items, concepts, responses = _full_sequences(batch)
        # Official simpleKT reconstructs its full sequence from a current and
        # shifted batch. Supply observed history responses plus the placeholder,
        # never the supervised next-response target tensor.
        official_batch = {
            "qseqs": items[:, :-1], "shft_qseqs": items[:, 1:],
            "cseqs": concepts[:, :-1], "shft_cseqs": concepts[:, 1:],
            "rseqs": responses[:, :-1], "shft_rseqs": responses[:, 1:],
        }
        return self.core(official_batch, return_logits=True)[:, 1:]
