import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_multiple_hist(res_list: list[tuple[list[float], str]], title: str = ""):
    fig, axes = plt.subplots(nrows=len(res_list), ncols=1, figsize=(8, 12), sharex=True)
    fig.subplots_adjust(top=0.8)
    for i, (data, label) in enumerate(res_list):
        sns.histplot(data, ax=axes[i], kde=False, bins=20, color='skyblue', edgecolor='black')
        axes[i].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(data):.2f}')
        axes[i].set_title(label)
        axes[i].set_ylabel("Count")
    axes[-1].set_xlabel("Value")
    plt.tight_layout()
    fig.suptitle(title, y=1.02)
    plt.show()