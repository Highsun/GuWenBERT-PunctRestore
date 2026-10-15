import torch
from transformers import BertConfig, BertForTokenClassification

from guwenbert_task1.common import LABELS


def test_real_token_classifier_backward_and_safetensors_reload(tmp_path):
    torch.manual_seed(42)
    model = BertForTokenClassification(BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=16, num_labels=len(LABELS)))
    inputs = {"input_ids": torch.tensor([[1, 4, 5, 2, 0, 0, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])}
    output = model(**inputs, labels=torch.tensor([[-100, 1, 2, -100, -100, -100, -100, -100]]))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    model.eval()
    with torch.inference_mode():
        reference = model(**inputs).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").exists()
    restored = BertForTokenClassification.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.inference_mode():
        assert torch.allclose(restored(**inputs).logits, reference)
