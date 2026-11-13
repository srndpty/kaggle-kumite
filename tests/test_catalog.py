import pytest

from kumite.catalog import find_competition, load_catalog
from kumite.paths import CATALOG_CSV


def test_catalog_has_100_unique_competitions() -> None:
    catalog = load_catalog(CATALOG_CSV)

    assert [c.number for c in catalog] == list(range(1, 101))
    assert len({c.slug for c in catalog}) == 100
    assert all(c.url.endswith(f"/competitions/{c.slug}") for c in catalog)


def test_dir_name_is_zero_padded_number_and_slug() -> None:
    first = find_competition(load_catalog(CATALOG_CSV), 1)

    assert first.dir_name == "001-otto-group-product-classification-challenge"


def test_find_competition_rejects_unknown_number() -> None:
    with pytest.raises(LookupError):
        find_competition(load_catalog(CATALOG_CSV), 101)
