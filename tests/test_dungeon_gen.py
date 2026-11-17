import pytest

from game.dungeon_gen import (
    split_bsp,
    rooms_overlap,
    connect_rooms,
    flood_fill,
    generate
)


def test_split_two():
    assert len(split_bsp((0, 0, 10, 10), 3, 1)) == 2

def test_split_min():
    assert all(r[2] >= 3 and r[3] >= 3 for r in split_bsp((0, 0, 10, 10), 3, 1))

def test_overlap_true():
    assert rooms_overlap((0, 0, 5, 5), (1, 1, 6, 6)) is True

def test_overlap_false():
    assert rooms_overlap((0, 0, 5, 5), (6, 6, 10, 10)) is False

def test_overlap_touching():
    assert rooms_overlap((0, 0, 5, 5), (5, 5, 10, 10)) is False

def test_connect_h():
    assert (1, 1) in connect_rooms((0, 0, 2, 2), (3, 0, 5, 2))

def test_connect_v():
    assert (1, 1) in connect_rooms((0, 0, 2, 2), (0, 3, 2, 5))

def test_flood_all():
    assert len(flood_fill([[1, 1], [1, 1]], (0, 0))) == 4

def test_flood_blocked():
    assert len(flood_fill([[1, 0], [0, 0]], (0, 0))) == 1

def test_generate_rooms():
    assert len(generate(1, 20, 20)) > 0

def test_generate_deterministic():
    assert generate(1, 20, 20) == generate(1, 20, 20)

def test_generate_diff_seed():
    assert generate(1, 20, 20) != generate(2, 20, 20)

def test_generate_empty():
    assert generate(1, 0, 0) == []

def test_generate_retry():
    assert generate(3, 10, 10) is not None

def test_flood_out():
    assert flood_fill([[0, 0], [0, 0]], (0, 0)) == []
