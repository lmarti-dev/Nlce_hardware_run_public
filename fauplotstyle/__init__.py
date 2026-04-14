"""Drop-in replacement for fauplotstyle using matplotlib built-in styles."""

import matplotlib.pyplot as plt


def style():
    plt.style.use("seaborn-v0_8-paper")
