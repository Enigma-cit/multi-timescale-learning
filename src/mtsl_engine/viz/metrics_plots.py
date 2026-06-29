import matplotlib.pyplot as plt


def plot_metrics(loss_list, acc_list, filename: str | None = None):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    steps = range(len(loss_list))
    ax1.plot(steps, loss_list, "C0-", label="loss")
    ax2.plot(steps, acc_list, "C1-", label="acc")

    ax1.set_xlabel("Eval step index")
    ax1.set_ylabel("Loss", color="C0")
    ax2.set_ylabel("Accuracy", color="C1")
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=200)
    else:
        plt.show()

    return fig