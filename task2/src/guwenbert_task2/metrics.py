from __future__ import annotations

from .common import LABELS


def calculate(gold_rows: list[list[int]], predicted_rows: list[list[int]], masks=None) -> dict:
    if len(gold_rows) != len(predicted_rows):
        raise ValueError("Record count mismatch")
    matrix = [[0] * 8 for _ in range(8)]
    for row_index, (gold, predicted) in enumerate(zip(gold_rows, predicted_rows)):
        if len(gold) != len(predicted):
            raise ValueError("Character count mismatch")
        for position, (g, p) in enumerate(zip(gold, predicted)):
            if masks is None or position in masks[row_index]:
                if not 0 <= g < 8 or not 0 <= p < 8:
                    raise ValueError("Invalid label")
                matrix[g][p] += 1

    def prf(tp, pn, gn):
        return {"precision": tp / pn if pn else 0.0, "recall": tp / gn if gn else 0.0,
                "f1": 2 * tp / (pn + gn) if pn + gn else 0.0, "support": gn, "predicted": pn}

    classes = {LABELS[i]: prf(matrix[i][i], sum(row[i] for row in matrix), sum(matrix[i])) for i in range(8)}
    tp = sum(matrix[i][i] for i in range(1, 8))
    pn = sum(sum(row[1:]) for row in matrix)
    gn = sum(sum(row) for row in matrix[1:])
    def boundaries(indices):
        return prf(sum(matrix[i][j] for i in indices for j in indices),
                   sum(matrix[i][j] for i in range(8) for j in indices),
                   sum(sum(matrix[i]) for i in indices))
    total = sum(map(sum, matrix))
    return {"characters": total, "punctuation_micro": prf(tp, pn, gn),
            "punctuation_macro_f1": sum(classes[label]["f1"] for label in LABELS[1:]) / 7,
            "segmentation_any_punctuation": boundaries(range(1, 8)),
            "sentence_boundary": boundaries([2, 3, 4]), "per_class": classes,
            "accuracy_including_O": sum(matrix[i][i] for i in range(8)) / total if total else 0.0,
            "confusion_gold_rows_predicted_columns": matrix}


def quality(rows: list[dict], inference: dict, margin: int) -> dict:
    gold = [row["labels"] for row in rows]
    predicted = inference["labels"]
    edge_masks = [set() for _ in rows]
    join_masks = [set() for _ in rows]
    for window in inference["windows"]:
        doc = window["document"]
        end = window["core_end"]
        if end < len(gold[doc]):
            edge_masks[doc].update(range(max(0, end - margin), min(len(gold[doc]), end + margin)))
    for doc, row in enumerate(rows):
        for end in row.get("joins", []):
            join_masks[doc].update(range(max(0, end - margin), min(len(gold[doc]), end + margin)))
    interior = [set(range(len(g))) - edges for g, edges in zip(gold, edge_masks)]
    return {"overall": calculate(gold, predicted), "window_edges": calculate(gold, predicted, edge_masks),
            "window_interiors": calculate(gold, predicted, interior),
            "reconstruction_joins": calculate(gold, predicted, join_masks),
            "exact_records": sum(g == p for g, p in zip(gold, predicted)), "boundary_margin_characters": margin}
