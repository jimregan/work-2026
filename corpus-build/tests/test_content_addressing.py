from datetime import date, datetime

from corpus_build.model.content_addressing import content_address, locator_address


def test_content_address_is_stable_for_identical_bytes() -> None:
    assert content_address(b"hello") == content_address(b"hello")


def test_content_address_differs_for_different_bytes() -> None:
    assert content_address(b"hello") != content_address(b"world")


def test_locator_address_differs_by_date_for_same_origin() -> None:
    origin = "https://example.org/broadcast.mp3"
    first = locator_address(origin, date(2026, 1, 1))
    second = locator_address(origin, date(2026, 1, 2))
    assert first != second


def test_locator_address_is_stable_for_same_origin_and_date() -> None:
    origin = "https://example.org/broadcast.mp3"
    at = datetime(2026, 1, 1, 12, 0, 0)
    assert locator_address(origin, at) == locator_address(origin, at)


def test_locator_address_differs_by_origin_for_same_date() -> None:
    at = date(2026, 1, 1)
    assert locator_address("https://example.org/a", at) != locator_address("https://example.org/b", at)
