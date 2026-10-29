import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fchgnn_repro.models import build_model
from fchgnn_repro.modern_baselines import _full_sequences


def batch(device="cpu"):
    values = {
        "item": [[0, 1, 2, 3, 4], [2, 3, 4, 0, 0]],
        "next_item": [[1, 2, 3, 4, 5], [3, 4, 5, 0, 0]],
        "concept": [[0, 1, 2, 3, 4], [1, 2, 3, 0, 0]],
        "next_concept": [[1, 2, 3, 4, 0], [2, 3, 4, 0, 0]],
        "correct": [[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        "target": [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]],
        "mask": [[True, True, True, True, True], [True, True, True, False, False]],
    }
    return {key: torch.tensor(value, device=device) for key, value in values.items()}


def model(name, device="cpu", dropout=0.0):
    torch.manual_seed(11)
    # Avoid coincidental embedding sizes triggering upstream reset's size test.
    return build_model(name, num_items=11, num_concepts=5, embedding_dim=16,
                       hidden_dim=32, num_heads=4, max_length=5,
                       dropout=dropout).to(device)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_target_labels_are_never_model_inputs(name):
    net = model(name).eval()
    original = batch()
    changed = {key: value.clone() for key, value in original.items()}
    changed["target"] = 1 - changed["target"]
    with torch.no_grad():
        expected = net(original)
        actual = net(changed)
    assert expected.shape == original["target"].shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_future_and_current_response_cannot_leak_to_predictions(name):
    net = model(name).eval()
    original = batch()
    changed = {key: value.clone() for key, value in original.items()}
    # Output index 2 predicts question 3, so response 3 must not influence it.
    changed["correct"][:, 3:] = 1 - changed["correct"][:, 3:]
    with torch.no_grad():
        expected = net(original)
        actual = net(changed)
    torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=1e-6, atol=1e-6)
    # Ensure the model actually uses historical responses (non-vacuous test).
    assert not torch.allclose(actual[0, 3:], expected[0, 3:], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_unscored_real_events_remain_observed_history(name):
    net = model(name).eval()
    original = batch()
    original["event_idx"] = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, -1, -1]])
    changed = {key: value.clone() for key, value in original.items()}
    changed["mask"][:, 0] = False
    changed["mask"][:, 2] = False
    with torch.no_grad():
        expected = net(original)
        actual = net(changed)
    # Supervision availability is not an input to the neural representation.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_trailing_padding_does_not_change_valid_predictions(name):
    net = model(name).eval()
    original = batch()
    changed = {key: value.clone() for key, value in original.items()}
    for key in ("item", "next_item", "concept", "next_concept"):
        changed[key][1, 3:] = 4
    changed["correct"][1, 3:] = 1
    with torch.no_grad():
        expected = net(original)
        actual = net(changed)
    torch.testing.assert_close(actual[1, :3], expected[1, :3], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_finite_masked_training_backward(name, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    net = model(name, device, dropout=0.2).train()
    data = batch(device)
    if name == "akt":
        loss, logits = net.training_objective(data)
    else:
        logits = net(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[data["mask"]], data["target"][data["mask"]])
    assert logits.device.type == device
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_official_sigmoid_and_exposed_logits_agree(name):
    net = model(name).eval()
    items, concepts, responses = _full_sequences(batch())
    with torch.no_grad():
        if name == "akt":
            probabilities, _ = net.core(concepts, responses, items)
            logits, _ = net.core(concepts, responses, items, return_logits=True)
        else:
            raw = {
                "qseqs": items[:, :-1], "shft_qseqs": items[:, 1:],
                "cseqs": concepts[:, :-1], "shft_cseqs": concepts[:, 1:],
                "rseqs": responses[:, :-1], "shft_rseqs": responses[:, 1:],
            }
            probabilities = net.core(raw)
            logits = net.core(raw, return_logits=True)
    torch.testing.assert_close(probabilities, logits.sigmoid())


def test_vendored_sources_match_recorded_checksums():
    root = Path(__file__).resolve().parents[1] / "src/fchgnn_repro/vendor/pykt"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    assert provenance["license"] == "MIT"
    for name, record in provenance["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == record["vendored_sha256"]
