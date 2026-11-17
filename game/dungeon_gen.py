"""Dungeon generation primitives.

Coordinate conventions
----------------------
Rectangles are ``(x, y, w, h)``: ``(x, y)`` is the top-left cell and ``w`` /
``h`` are the number of cells along each axis.  Cell ``(cx, cy)`` belongs to a
rectangle when ``x <= cx < x + w`` and ``y <= cy < y + h``.  Rectangles are
half-open, so two rectangles that merely touch along an edge do **not**
overlap.

Fixes implemented here
----------------------
* Rooms are split with a BSP that respects both a minimum room size and a
  maximum aspect ratio, and split positions are always clamped inside the
  parent rectangle.
* ``connect_rooms`` returns a contiguous corridor whose end points are doors
  carved into the facing walls; when the rooms overlap on the perpendicular
  axis both doors share the same row/column, so the corridor is aligned.
* Integer, string and byte seeds are all reduced deterministically, so the
  same seed always produces the same level (replays / daily challenges).
* ``rooms_overlap`` works on exact cell coverage, including L-shaped rooms
  made from several rectangles or an explicit cell set.
* ``flood_fill`` walks every cell, so a corridor that is broken in the middle
  is detected instead of only checking room centres.
* ``generate`` retries a bounded number of times and never loops forever.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque

__all__ = [
    "split_bsp",
    "rooms_overlap",
    "connect_rooms",
    "flood_fill",
    "generate",
]

MIN_ROOM_SIZE = 3
MAX_ASPECT_RATIO = 3.0
MAX_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clamp(value, low, high):
    """Clamp ``value`` into ``[low, high]`` (``low`` wins if the range is bad)."""
    if low > high:
        return low
    return max(low, min(high, value))


def _center(rect):
    x, y, w, h = rect[:4]
    return int(x + w // 2), int(y + h // 2)


def _hspan(y, x_from, x_to):
    """Inclusive, contiguous horizontal run of cells (left-to-right order)."""
    step = 1 if x_to >= x_from else -1
    return [(x, y) for x in range(x_from, x_to + step, step)]


def _vspan(x, y_from, y_to):
    step = 1 if y_to >= y_from else -1
    return [(x, y) for y in range(y_from, y_to + step, step)]


def _dedupe_consecutive(cells):
    result = []
    for cell in cells:
        if not result or result[-1] != cell:
            result.append(cell)
    return result


def _seed_to_int(seed):
    """Deterministically map any common seed type to an integer.

    ``random.Random(str)`` uses Python's per-process randomized ``hash()`` for
    strings, which would break replay.  Hashing explicitly with SHA-256 keeps
    string seeds stable across processes and runs.
    """
    if isinstance(seed, int):  # bool is an int subclass; that is fine
        return int(seed)
    if isinstance(seed, bytes):
        data = seed
    else:
        data = str(seed).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _attempt_seed(base_seed, attempt):
    """Derive a stable RNG seed for a retry attempt."""
    return (int(base_seed) + int(attempt) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


# ---------------------------------------------------------------------------
# BSP splitting
# ---------------------------------------------------------------------------

def _cut_range(rect, axis, min_size, max_aspect):
    """Feasible split positions for ``axis``, or ``None``.

    A split must leave ``min_size`` cells on both sides.  When ``max_aspect``
    is finite it additionally rejects slices whose resulting aspect ratio would
    exceed the limit.  The returned range is inclusive.
    """
    _, _, w, h = rect[:4]
    if axis == "v":
        long_side, fixed = w, h
    else:
        long_side, fixed = h, w
    if long_side < 2 * min_size:
        return None

    low = min_size
    high = long_side - min_size

    if max_aspect is not None and math.isfinite(max_aspect) and max_aspect > 0:
        # Both pieces need   c >= fixed / A   and   c <= A * fixed.
        low = max(low, math.ceil(fixed / max_aspect))
        high = min(high, int(math.floor(max_aspect * fixed)))
        # The other piece has the same aspect constraints.
        low = max(low, long_side - int(math.floor(max_aspect * fixed)))
        high = min(high, long_side - math.ceil(fixed / max_aspect))

    low = max(low, 1)
    high = min(high, long_side - 1)
    if low > high:
        return None
    return int(low), int(high)


def _best_effort_cut(rect, axis, min_size):
    """Pick the least elongated min-size split when the aspect limit is
    mathematically unreachable (for example a 7x60 strip whose short side
    cannot be halved).  Ties are broken towards an even split, so repeated
    subdivision still converges towards compact rooms."""
    cut_range = _cut_range(rect, axis, min_size, None)
    if cut_range is None:
        return None
    _, _, w, h = rect[:4]
    long_side = w if axis == "v" else h
    fixed = h if axis == "v" else w
    half = long_side // 2

    def penalty(cut):
        return (max(cut, long_side - cut) / fixed, abs(cut - half))

    return min(range(cut_range[0], cut_range[1] + 1), key=penalty)


def _split_leaf(rect, min_size, rng, max_aspect):
    """Split one BSP leaf once, or return it unchanged when impossible."""
    x, y, w, h = rect[:4]
    if w < min_size or h < min_size:
        # No leaf inside this parent can honour the minimum size on both axes.
        return [rect] if (w <= 0 or h <= 0) else []
    can_vertical = w >= 2 * min_size
    can_horizontal = h >= 2 * min_size
    if not can_vertical and not can_horizontal:
        return [rect]

    # Prefer splitting the longer axis: this is what prevents long, thin
    # strips from being produced by an area-only split.
    if can_vertical and can_horizontal:
        if w >= h * 1.25:
            axis = "v"
        elif h >= w * 1.25:
            axis = "h"
        else:
            axis = "v" if rng.random() < 0.5 else "h"
    else:
        axis = "v" if can_vertical else "h"

    cut_range = _cut_range(rect, axis, min_size, max_aspect)
    if cut_range is None:
        other = "h" if axis == "v" else "v"
        other_allowed = (other == "v" and can_vertical) or (
            other == "h" and can_horizontal
        )
        if other_allowed:
            axis = other
            cut_range = _cut_range(rect, axis, min_size, max_aspect)

    if cut_range is not None:
        cut = rng.randint(*cut_range)
    else:
        # The aspect limit is unreachable for this parent.  Split anyway, at
        # the cut that brings the pieces as close to square as possible;
        # refusing to split would leave one very long room.
        cut = _best_effort_cut(rect, axis, min_size)
        if cut is None:
            other = "h" if axis == "v" else "v"
            if (other == "v" and can_vertical) or (
                other == "h" and can_horizontal
            ):
                cut = _best_effort_cut(rect, other, min_size)
                if cut is not None:
                    axis = other
        if cut is None:
            return [rect]

    if axis == "v":
        return [(x, y, cut, h), (x + cut, y, w - cut, h)]
    return [(x, y, w, cut), (x, y + cut, w, h - cut)]


def split_bsp(rect, min_size=MIN_ROOM_SIZE, depth=1, rng=None,
              max_aspect=MAX_ASPECT_RATIO):
    """Split ``rect`` with binary space partitioning.

    Parameters
    ----------
    rect:
        ``(x, y, w, h)`` region to split.
    min_size:
        Minimum width and height of every returned leaf.
    depth:
        Number of subdivision rounds.  ``depth=1`` produces at most two leaves,
        ``depth=2`` at most four, and so on.
    rng:
        Optional ``random.Random`` instance.  When omitted a fixed seed is used
        so that results are repeatable.
    max_aspect:
        Maximum ``max(w, h) / min(w, h)`` a resulting piece may have, when a
        split along the other axis is possible.

    Splits always stay inside the parent rectangle; a leaf that cannot be
    halved while keeping both sides at ``min_size`` is returned unchanged.
    """
    x, y, w, h = rect[:4]
    if w <= 0 or h <= 0 or min_size <= 0:
        return []
    if w < min_size or h < min_size:
        return []
    if rng is None:
        rng = random.Random(0)

    leaves = [(int(x), int(y), int(w), int(h))]
    for _ in range(max(0, int(depth))):
        next_leaves = []
        for leaf in leaves:
            next_leaves.extend(_split_leaf(leaf, int(min_size), rng, max_aspect))
        leaves = next_leaves
    return leaves


# ---------------------------------------------------------------------------
# Precise overlap (including L-shaped rooms)
# ---------------------------------------------------------------------------

def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_rects(room):
    """Normalise a room description into a list of ``(x, y, w, h)`` rects.

    Supported forms:

    * a rectangle ``(x, y, w, h)``;
    * an explicit cell ``(x, y)``;
    * a sequence of rectangles (used for L-shaped rooms);
    * a set/sequence of cells;
    * a mapping with ``rects``, ``cells``, ``x/y/w/h`` or ``x1/y1/x2/y2``.

    Rectangles are exact: an L-shaped room made of two rectangles only covers
    the cells of those two rectangles, not their bounding box.
    """
    if isinstance(room, dict):
        if "cells" in room:
            return [(int(cx), int(cy), 1, 1) for cx, cy in room["cells"]]
        if "rects" in room:
            rects = []
            for item in room["rects"]:
                rects.extend(_as_rects(item))
            return rects
        if all(key in room for key in ("x", "y", "w", "h")):
            return [(int(room["x"]), int(room["y"]),
                     int(room["w"]), int(room["h"]))]
        if all(key in room for key in ("x1", "y1", "x2", "y2")):
            x1, y1 = int(room["x1"]), int(room["y1"])
            return [(x1, y1, int(room["x2"]) - x1, int(room["y2"]) - y1)]
        raise TypeError("unsupported room mapping")

    if isinstance(room, (set, frozenset)):
        return [(int(cx), int(cy), 1, 1) for cx, cy in room]

    if isinstance(room, (tuple, list)):
        if len(room) == 4 and all(_is_number(v) for v in room):
            return [(int(room[0]), int(room[1]),
                     int(room[2]), int(room[3]))]
        if len(room) == 2 and all(_is_number(v) for v in room):
            return [(int(room[0]), int(room[1]), 1, 1)]
        rects = []
        for item in room:
            rects.extend(_as_rects(item))
        return rects

    raise TypeError("unsupported room description: %r" % (room,))


def _rects_intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        aw > 0 and ah > 0 and bw > 0 and bh > 0
        and ax < bx + bw and bx < ax + aw
        and ay < by + bh and by < ay + ah
    )


def rooms_overlap(room_a, room_b):
    """Return ``True`` when the two rooms share at least one cell.

    The comparison uses exact occupied cells, not bounding boxes.  Rooms may be
    described by several rectangles (L-shapes) or by explicit cell sets; see
    :func:`_as_rects`.  Edge-touching rooms do not overlap.
    """
    rects_a = _as_rects(room_a)
    rects_b = _as_rects(room_b)
    for rect_a in rects_a:
        for rect_b in rects_b:
            if _rects_intersect(rect_a, rect_b):
                return True
    return False


# ---------------------------------------------------------------------------
# Corridors / doors
# ---------------------------------------------------------------------------

def connect_rooms(room_a, room_b):
    """Carve a contiguous corridor between two ``(x, y, w, h)`` rooms.

    The returned list starts at a door on the facing wall of ``room_a`` and
    ends at a door on the facing wall of ``room_b``.  Both doors are chosen on
    the same row (horizontal connection) or column (vertical connection)
    whenever the rooms overlap on the perpendicular axis, so the corridor
    always meets the opening instead of running into a solid wall.

    When the rooms do not overlap on the perpendicular axis the corridor bends
    once and still enters each room through its chosen door.
    """
    ax, ay, aw, ah = room_a[:4]
    bx, by, bw, bh = room_b[:4]
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ca = _center(room_a)
    cb = _center(room_b)

    if abs(cb[0] - ca[0]) >= abs(cb[1] - ca[1]):
        # Horizontal connection: doors are vertical wall cells.
        if cb[0] >= ca[0]:
            door_a_x = ax2 - 1
            door_b_x = bx1
        else:
            door_a_x = ax1
            door_b_x = bx2 - 1

        overlap_low = max(ay1, by1)
        overlap_high = min(ay2, by2) - 1
        if overlap_low <= overlap_high:
            door_y = _clamp((ca[1] + cb[1]) // 2, overlap_low, overlap_high)
            door_a_y = door_b_y = door_y
        else:
            mid_y = (ca[1] + cb[1]) // 2
            door_a_y = _clamp(mid_y, ay1, ay2 - 1)
            door_b_y = _clamp(mid_y, by1, by2 - 1)

        door_a = (int(door_a_x), int(door_a_y))
        door_b = (int(door_b_x), int(door_b_y))
        bend_x = (door_a[0] + door_b[0]) // 2

        path = _hspan(door_a[1], door_a[0], bend_x)
        path += _vspan(bend_x, door_a[1], door_b[1])
        path += _hspan(door_b[1], bend_x, door_b[0])
    else:
        # Vertical connection: doors are horizontal wall cells.
        if cb[1] >= ca[1]:
            door_a_y = ay2 - 1
            door_b_y = by1
        else:
            door_a_y = ay1
            door_b_y = by2 - 1

        overlap_low = max(ax1, bx1)
        overlap_high = min(ax2, bx2) - 1
        if overlap_low <= overlap_high:
            door_x = _clamp((ca[0] + cb[0]) // 2, overlap_low, overlap_high)
            door_a_x = door_b_x = door_x
        else:
            mid_x = (ca[0] + cb[0]) // 2
            door_a_x = _clamp(mid_x, ax1, ax2 - 1)
            door_b_x = _clamp(mid_x, bx1, bx2 - 1)

        door_a = (int(door_a_x), int(door_a_y))
        door_b = (int(door_b_x), int(door_b_y))
        bend_y = (door_a[1] + door_b[1]) // 2

        path = _vspan(door_a[0], door_a[1], bend_y)
        path += _hspan(bend_y, door_a[0], door_b[0])
        path += _vspan(door_b[0], bend_y, door_b[1])

    return _dedupe_consecutive(path)


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

def flood_fill(grid, start):
    """Return every cell reachable from ``start`` through walkable cells.

    Walkable cells are truthy values; movement is four-directional.  The start
    must be inside the grid and walkable, otherwise ``[]`` is returned.  The
    result is in deterministic breadth-first order.
    """
    if not grid:
        return []
    try:
        sx, sy = int(start[0]), int(start[1])
    except (TypeError, ValueError, IndexError):
        return []

    height = len(grid)
    if sy < 0 or sy >= height:
        return []
    if sx < 0 or sx >= len(grid[sy]):
        return []
    if not grid[sy][sx]:
        return []

    seen = {(sx, sy)}
    queue = deque([(sx, sy)])
    visited = []

    while queue:
        x, y = queue.popleft()
        visited.append((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if ny < 0 or ny >= height:
                continue
            if nx < 0 or nx >= len(grid[ny]):
                continue
            if (nx, ny) in seen or not grid[ny][nx]:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))

    return visited


# ---------------------------------------------------------------------------
# Level generation
# ---------------------------------------------------------------------------

def _rooms_from_leaves(leaves, min_room_size):
    """Inset BSP leaves by one wall cell and keep rooms large enough."""
    rooms = []
    for x, y, w, h in leaves:
        if w >= min_room_size + 2 and h >= min_room_size + 2:
            room = (x + 1, y + 1, w - 2, h - 2)
        elif w >= min_room_size and h >= min_room_size:
            room = (x, y, w, h)
        else:
            continue
        if room not in rooms:
            rooms.append(room)
    return rooms


def _build_grid(width, height, rooms):
    """Rasterise rooms and their corridors into a wall/floor grid."""
    grid = [[0] * width for _ in range(height)]

    for x, y, w, h in rooms:
        for cy in range(max(0, y), min(height, y + h)):
            row = grid[cy]
            for cx in range(max(0, x), min(width, x + w)):
                row[cx] = 1

    for room_a, room_b in zip(rooms, rooms[1:]):
        for cx, cy in connect_rooms(room_a, room_b):
            if 0 <= cx < width and 0 <= cy < height:
                grid[cy][cx] = 1

    return grid


def _connectivity_score(grid):
    """Return ``(reachable_floor_cells, total_floor_cells)`` for a layout."""
    floor = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value:
                floor.append((x, y))
    if not floor:
        return 0, 0
    return len(flood_fill(grid, floor[0])), len(floor)


def _depth_for(width, height, split_min):
    """Choose a bounded subdivision depth from the map size."""
    longest = max(width, height)
    if longest < split_min or split_min <= 0:
        return 1
    rounds = int(math.log2(longest / split_min)) + 1
    return max(1, min(5, rounds))


def generate(seed, width, height, min_room_size=MIN_ROOM_SIZE,
             max_attempts=MAX_ATTEMPTS):
    """Generate and return a deterministic list of ``(x, y, w, h)`` rooms.

    The same ``seed`` always produces the same layout.  ``seed`` may be an
    integer or text (for example a daily-challenge key); text is hashed with
    SHA-256 so it is stable across processes.

    Each attempt builds rooms with ``split_bsp`` (minimum size and aspect-ratio
    constrained), aligns doors and carves corridors with ``connect_rooms``, and
    then validates the layout with a full-cell ``flood_fill``.  A failed
    attempt is retried at most ``max_attempts`` times; the best partial layout
    is returned rather than looping forever.  Invalid or empty maps return
    ``[]``.
    """
    width = int(width)
    height = int(height)
    min_room_size = int(min_room_size)
    if width <= 0 or height <= 0:
        return []
    if width < min_room_size or height < min_room_size:
        return []

    base_seed = _seed_to_int(seed)
    # Leaves need two wall cells to become a bordered room.  On narrow maps
    # fall back to the bare minimum so a single strip of rooms can still be
    # generated, instead of returning an empty level.
    split_min = min(int(min_room_size) + 2, width, height)
    depth = _depth_for(width, height, split_min)
    attempts = max(1, int(max_attempts))

    best_rooms = None
    best_score = (-1, -1)

    for attempt in range(attempts):
        rng = random.Random(_attempt_seed(base_seed, attempt))
        leaves = split_bsp(
            (0, 0, width, height),
            min_size=split_min,
            depth=depth,
            rng=rng,
            max_aspect=MAX_ASPECT_RATIO,
        )
        rooms = _rooms_from_leaves(leaves, int(min_room_size))
        if not rooms:
            continue

        grid = _build_grid(width, height, rooms)
        reachable, total = _connectivity_score(grid)
        if total and reachable == total:
            return rooms

        # Keep the most connected layout as a bounded fallback.
        if (reachable, total) > best_score:
            best_score = (reachable, total)
            best_rooms = rooms

    return best_rooms if best_rooms is not None else []
