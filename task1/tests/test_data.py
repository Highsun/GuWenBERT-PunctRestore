import pytest

from guwenbert_task1.common import LABELS, TAG_TO_ID, render
from guwenbert_task1.data import clean_training, normalize_input, parse_conll


def test_native_tag_order_and_after_character_alignment(tmp_path):
    source = tmp_path / "train.txt"
    source.write_text('-DOCSTART- -SEG- -PUNC- -QUOTE- -BOOK-\n\n子 O O O O\n曰 B B-: O O\n学 O O B O\n乎 B B-? I O\n', encoding='utf-8')
    rows = parse_conll(source, 'train')
    assert len(rows) == 1
    assert rows[0]['text'] == '子曰学乎'
    assert render(rows[0]['text'], rows[0]['labels']) == '子曰：学乎？'
    assert LABELS[TAG_TO_ID['B-\\']] == '、'
    assert rows[0]['quote_tags'] == ['O', 'O', 'B', 'I']


def test_empty_records_and_document_boundaries_do_not_create_samples(tmp_path):
    source = tmp_path / 'dev.txt'
    source.write_text('-DOCSTART-\n\n\n甲 O O O O\n\n\n-DOCSTART-\n乙 B B-. O O\n\n')
    rows = parse_conll(source, 'validation')
    assert [row['text'] for row in rows] == ['甲', '乙']
    assert [row['document_index'] for row in rows] == [0, 1]


@pytest.mark.parametrize('line', ['甲 O O O', '甲 O BAD O O', '甲 B O O O', '甲乙 O O O O', '甲 O O E O'])
def test_invalid_native_annotation_is_rejected(tmp_path, line):
    source = tmp_path / 'bad.txt'
    source.write_text(line + '\n')
    with pytest.raises(ValueError):
        parse_conll(source, 'train')


def test_holdout_is_preserved_and_only_training_duplicates_are_filtered():
    row = lambda identity, digest: {'article_id': identity, 'content_sha256': digest}
    validation, test = [row('dev', 'shared')], [row('test', 'other')]
    train = [row('a', 'shared'), row('b', 'unique'), row('c', 'unique'), row('d', 'other')]
    kept, removed = clean_training(train, validation, test)
    assert [item['article_id'] for item in kept] == ['b']
    assert [item['reason'] for item in removed] == ['exact_text_in_holdout', 'duplicate_train_text', 'exact_text_in_holdout']
    assert validation == [row('dev', 'shared')] and test == [row('test', 'other')]


def test_demo_normalization_does_not_rewrite_nontarget_characters():
    clean, audit = normalize_input('“子曰：学乎？” 3.14％\x16')
    assert clean == '子曰学乎3.14％'
    assert audit['removed:U+0016'] == 1
