# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Keyboard layouts of the fly / roll / view keys (QWERTY, AZERTY)."""

from PyQt6.QtCore import Qt

# fly / roll keys at the same physical positions on both layouts
KEYMAPS = {
    "qwerty": {
        "fly": {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, -1, 0), "d": (0, 1, 0), "r": (0, 0, 1), "f": (0, 0, -1)},
        "roll": {"q": 1, "e": -1},
        "views": {},
    },
    "azerty": {
        "fly": {"z": (1, 0, 0), "s": (-1, 0, 0), "q": (0, -1, 0), "d": (0, 1, 0), "r": (0, 0, 1), "f": (0, 0, -1)},
        "roll": {"a": 1, "e": -1},
        "views": {"&": "1", '"': "3", "è": "7"},  # unshifted digit row
    },
}


# view / navigate mode (gui/viewmode.py): action -> shortcuts, the same on both layouts (characters, not
# positions). Space / B step through the dumps as in splash; no letter here is a fly / roll key of either layout.
VIEW_KEYS = {
    "next": ["Space", "Right"],
    "prev": ["B", "Left"],
    "next10": ["Shift+Space", "Shift+Right"],
    "prev10": ["Shift+B", "Shift+Left"],
    "first": ["Home"],
    "last": ["End"],
    "play": ["P"],
    "zoom_in": ["+", "="],
    "zoom_out": ["-"],
    "auto_clim": ["C"],
    "recentre": ["O"],
    "look_at": ["T"],
    "ride": ["Shift+T"],
    "grid": ["G"],
    "help": ["H"],
    "deselect": ["Escape"],
    "save_image": ["Ctrl+E"],
}
AXIS_VIEWS = ("1", "3", "7")  # front (from -Y), side (from +X), top (from +Z), as in the world view


def view_shortcuts(name):
    """VIEW_KEYS plus the axis views ("view1", "view3", "view7") for a keyboard layout."""
    keys = {k: list(v) for k, v in VIEW_KEYS.items()}
    for digit in AXIS_VIEWS:
        keys["view" + digit] = [digit] + [c for c, d in KEYMAPS[name]["views"].items() if d == digit]
    return keys


def keymap_hint(name):
    return "ZQSD/RF fly · A/E roll" if name == "azerty" else "WASD/RF fly · Q/E roll"


def key_char(ev):
    """Layout independent letter of a key event (Qt.Key follows the active keyboard layout)."""
    k = ev.key()
    if Qt.Key.Key_A.value <= k <= Qt.Key.Key_Z.value:
        return chr(k).lower()
    return ev.text().lower()
