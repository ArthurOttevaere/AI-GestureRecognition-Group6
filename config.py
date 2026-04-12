"""
Configuration – paths and constants shared across all modules.
"""

import os

_BASE = os.path.dirname(os.path.abspath(__file__))

DOMAIN1_DIR = os.path.join(_BASE, "GestureData_Mons",
                            "GestureDataDomain1_Mons", "Domain1_csv")
DOMAIN4_DIR = os.path.join(_BASE, "GestureData_Mons",
                            "GestureDataDomain4_Mons")

# Directory where all generated files (PNGs, CSVs) will be saved
DATA_DIR = os.path.join(_BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Human-readable labels for Domain 4 gesture classes (class id 1–10)
DOMAIN4_CLASS_NAMES = {
    1: "Cuboid",      2: "Cylinder",   3: "Sphere",
    4: "Rect. Pipe",  5: "Hemisphere", 6: "Cyl. Pipe",
    7: "Pyramid",     8: "Tetrahedron",9: "Cone",
    10: "Toroid",
}
