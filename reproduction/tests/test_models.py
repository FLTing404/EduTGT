import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fchgnn_repro.models import build_model
from fchgnn_repro.train import best_threshold, imbalance_loss


def batch():
    return {
        "item": torch.tensor([[0, 1, 2, 0]]),
        "concept": torch.tensor([[0, 0, 1, 0]]),
        "correct": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "next_item": torch.tensor([[1, 2, 3, 0]]),
        "next_concept": torch.tensor([[0, 1, 1, 0]]),
        "target": torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        "time_gap": torch.tensor([[0.0, 0.2, 0.4, 0.0]]),
        "next_time_gap": torch.tensor([[0.2, 0.4, 0.8, 0.0]]),
        "mask": torch.tensor([[True, True, True, False]]),
    }


def test_neural_shapes():
    common = dict(
        num_items=4,
        num_concepts=2,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        num_heads=2,
        max_length=4,
        hypergraph_layers=2,
        edge_item=torch.tensor([0, 1, 2, 3]),
        edge_concept=torch.tensor([0, 0, 1, 1]),
        difficulty=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    for name in ("dkt", "saint", "fchgnn"):
        output = build_model(name, **common)(batch())
        assert output.shape == (1, 4)
        assert torch.isfinite(output).all()


def test_advanced_fchgnn_objective_and_diffusion():
    model = build_model(
        "fchgnn",
        num_items=4,
        num_concepts=2,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        hypergraph_layers=2,
        edge_item=torch.tensor([0, 1, 2, 3]),
        edge_concept=torch.tensor([0, 0, 1, 1]),
        difficulty=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        time_aware=True,
        multi_concept=True,
        hypergraph_attention=True,
        dynamic_scale_fusion=True,
        contrastive_weight=0.01,
        uncertainty_enabled=True,
        guided_diffusion=True,
        diffusion_steps=3,
    )
    model.train()
    loss, logits = model.training_objective(batch(), epoch=1, total_epochs=2)
    assert logits.shape == (1, 4)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        prediction = model(batch())
        uncertainty = model.predictive_uncertainty(batch())
    assert prediction.shape == (1, 4)
    assert uncertainty is not None and uncertainty.shape == (1, 4)
    assert torch.isfinite(prediction).all()
    assert ((uncertainty > 0.0) & (uncertainty <= 1.0)).all()


def test_imbalance_losses_and_thresholds():
    logits = torch.tensor([3.0, 2.0, -0.5, -2.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 0.0, 1.0])
    for mode in ("weighted_bce", "focal", "auc_margin"):
        loss = imbalance_loss(logits, target, mode, negative_weight=4.0)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    y_true = target.numpy()
    y_prob = torch.sigmoid(logits.detach()).numpy()
    for objective in ("f1", "mcc", "balanced_accuracy", "incorrect_f1"):
        threshold = best_threshold(y_true, y_prob, objective)
        assert 0.05 <= threshold <= 0.95
