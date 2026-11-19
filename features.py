from __future__ import annotations

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 15
HAND_VECTOR_LENGTH = 42  # 21 landmarks * x/y
FRAME_VECTOR_LENGTH = 84 # left + right


def sequence_to_row(sequence: list[list[float]], label: str) -> dict:
    """
    Convert one 15-frame sequence into the same flat CSV format used by sign_language.csv.
    Each frame must contain 84 values: 42 left-hand coords + 42 right-hand coords.
    """
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} frames, got {len(sequence)}.")

    row = {}
    for frame_num, frame in enumerate(sequence):
        if len(frame) != FRAME_VECTOR_LENGTH:
            raise ValueError(f"Frame {frame_num} must have {FRAME_VECTOR_LENGTH} values.")

        landmark_num = 0
        for location_num, value in enumerate(frame):
            handedness = "L" if location_num < HAND_VECTOR_LENGTH else "R"
            coordinate = "x" if location_num % 2 == 0 else "y"
            column_name = f"{frame_num}, {handedness}, {landmark_num}, {coordinate}"
            row[column_name] = value

            if location_num % 2 != 0:
                landmark_num += 1
            if landmark_num > 20:
                landmark_num = 0

    row["label"] = label
    return row


def sequences_to_dataframe(sequences: list[list[list[float]]], label: str) -> pd.DataFrame:
    rows = [sequence_to_row(sequence, label) for sequence in sequences]
    return pd.DataFrame(rows)


def landmarks_to_frame_vector(landmarks) -> list[int]:
    """
    Convert detector output into one 84-number frame vector.
    Missing hands are filled with zeroes.
    """
    left_hand = [0] * HAND_VECTOR_LENGTH
    right_hand = [0] * HAND_VECTOR_LENGTH

    for hand in landmarks:
        handedness = hand[0]
        lmlist = hand[1]
        location_vector = [coord for lm in lmlist for coord in lm[1:3]]

        # MediaPipe labels are mirrored when the camera image is flipped.
        if handedness == "Left":
            left_hand = location_vector
        elif handedness == "Right":
            right_hand = location_vector

    return left_hand + right_hand


def dataframe_to_xy(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def feature_columns():
    return [f"{f}, {h}, {p}, {a}" for f in range(SEQUENCE_LENGTH)
            for h in ("L", "R") for p in range(21) for a in ("x", "y")]
