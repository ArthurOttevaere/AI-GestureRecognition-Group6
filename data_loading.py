"""
1. DATA LOADING
===============
Functions to load Domain 1 and Domain 4 gesture datasets.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

from config import DOMAIN4_CLASS_NAMES


def load_domain1(folder_path: str) -> tuple[list, list, list]:
    """
    Load all Domain 1 CSV files.

    Naming convention: SubjectS-G-R.csv
        S = subject index (1–10 in filename, stored as 0–9)
        G = gesture class (0–9, i.e. the digit drawn)
        R = repetition    (1–10)

    Returns
    -------
    data   : list of np.ndarray of shape (T, 3) – x, y, z coordinates
    labels : list of int – gesture class (0–9)
    users  : list of int – subject index (0–9)
    """
    data, labels, users = [], [], []
    pattern = re.compile(r"Subject(\d+)-(\d+)-(\d+)\.csv", re.IGNORECASE)

    for filename in sorted(os.listdir(folder_path)):
        m = pattern.match(filename)
        if m is None:
            continue

        subject = int(m.group(1)) - 1   # 1-based → 0-based
        gesture = int(m.group(2))

        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath, header=0, names=["x", "y", "z", "t"])
        coords = df[["x", "y", "z"]].values.astype(float)

        data.append(coords)
        labels.append(gesture)
        users.append(subject)

    return data, labels, users


def load_domain4(folder_path: str) -> tuple[list, list, list]:
    """
    Load all Domain 4 plain-text files (no extension).

    File format:
        Domain id = 4
        Class id  = C
        User id   = U   (1-based in file, stored as 0-based)
        <blank line>
        <x>,<y>,<z>,<t>
        ...

    Returns
    -------
    data   : list of np.ndarray of shape (T, 3)
    labels : list of int – gesture class (1–10)
    users  : list of int – subject index (0–9)
    """
    data, labels, users = [], [], []

    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, PermissionError):
            continue

        gesture, subject, data_start = None, None, 0

        for i, line in enumerate(lines):
            s = line.strip()
            if s.lower().startswith("class id"):
                gesture = int(s.split("=")[1].strip())
            elif s.lower().startswith("user id"):
                subject = int(s.split("=")[1].strip()) - 1
            elif s.lower().startswith("<x>"):
                data_start = i + 1
                break

        if gesture is None or subject is None:
            continue

        rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

        if not rows:
            continue

        data.append(np.array(rows, dtype=float))
        labels.append(gesture)
        users.append(subject)

    return data, labels, users


def check_completeness(labels: list, users: list, domain_name: str) -> None:
    """Verify that every (user, gesture) pair has exactly 10 repetitions."""
    counts: dict = defaultdict(int)
    for u, g in zip(users, labels):
        counts[(u, g)] += 1

    issues = [(u, g, n) for (u, g), n in counts.items() if n != 10]
    if issues:
        print(f"  [WARNING] {domain_name} – incomplete groups:")
        for u, g, n in sorted(issues):
            print(f"    user={u}, gesture={g} → {n} rep(s) found (expected 10)")
    else:
        print(f"  {domain_name}: {len(labels)} sequences loaded – completeness OK "
              f"({len(set(users))} users × {len(set(labels))} gestures × 10 reps)")


def print_dataset_info(data: list, labels: list, users: list,
                       domain_name: str) -> None:
    """Print basic statistics about a loaded dataset."""
    lengths = [len(seq) for seq in data]
    print(f"\n{'─'*50}")
    print(f"  {domain_name}")
    print(f"{'─'*50}")
    print(f"  Total sequences : {len(data)}")
    print(f"  Subjects        : {sorted(set(users))}")
    print(f"  Gesture classes : {sorted(set(labels))}")
    print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
    check_completeness(labels, users, domain_name)
