import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_record_classes_are_not_authenticated_cuff_cycles():
    with (ROOT / 'outputs/tables/Supp_Table_S07.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    classes = {row['Record class']: row for row in rows}
    assert int(classes['Same-value timer record']['Records, n']) == 53653
    assert int(classes['Total']['Records, n']) == 74027
    assert sum(int(row['Records, n']) for row in rows[:-1]) == 74027
    assert 'Not independent cuff measurements' in classes['Total']['Meaning']


def test_paired_record_total_retains_record_level_denominator():
    with (ROOT / 'outputs/tables/Supp_Table_S08.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    classes = {row['Paired record class']: row for row in rows}
    assert int(classes['Same-value timer']['Records, n']) == 48885
    assert int(classes['Total']['Records, n']) == 58842
    assert sum(int(row['Records, n']) for row in rows[:-1]) == 58842
