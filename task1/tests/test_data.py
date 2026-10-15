import pytest

from guwenbert_task1.common import LABEL_TO_ID, render
from guwenbert_task1.data import normalize_text, parse_line, split_articles


def test_compound_annotations_and_slashes_are_parsed_without_leaking_tags():
    identity, text = parse_line("19980101-01-001-001/m  [中国/ns  人民/n]nt  说/v  ：/w  ３/m  ／/w  ４/m")
    assert identity == "19980101-01-001"
    assert text == "中国人民说：３／４"


@pytest.mark.parametrize("line", ["bad/m  文字/n", "19980101-01-001-001/m  没有标注", "19980101-01-001-001/m  空标签/"])
def test_malformed_annotation_is_rejected(line):
    with pytest.raises(ValueError):
        parse_line(line)


def test_normalization_handles_sequence_targets_decimals_and_control_audit():
    text, labels, audit = normalize_text("“你好？！”他说：3.14。闭\x16幕。")
    assert text == "你好他说3.14闭幕"
    assert render(text, labels) == "你好！他说：3.14。闭幕。"
    assert labels[1] == LABEL_TO_ID["！"]
    assert audit["collapsed_target_sequences"] == 1
    assert audit["removed_control_or_space:U+0016"] == 1


def test_sentence_terminator_takes_precedence_over_following_comma():
    text, labels, audit = normalize_text("。你好。，世界")
    assert render(text, labels) == "你好。世界"
    assert audit["leading_target_punctuation"] == 1


def test_numeric_semantics_are_preserved_across_character_widths():
    text, labels, _ = normalize_text("增长１３．５％，占3/4。")
    assert text == "增长１３．５％占3/4"
    assert render(text, labels) == "增长１３．５％，占3/4。"


def test_split_is_reproducible_and_article_disjoint():
    rows = [{"article_id": str(index), "content_sha256": str(index)} for index in range(30)]
    first = split_articles(rows, 42, 0.8, 0.1)
    assert first == split_articles(list(reversed(rows)), 42, 0.8, 0.1)
    assert [len(rows) for rows in first.values()] == [24, 3, 3]
    ids = [{row["article_id"] for row in rows} for rows in first.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_duplicate_content_cannot_cross_splits():
    rows = [{"article_id": str(index), "content_sha256": "same"} for index in range(30)]
    with pytest.raises(ValueError, match="Duplicate"):
        split_articles(rows, 42, 0.8, 0.1)
