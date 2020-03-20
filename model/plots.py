from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from settings.dev import OUTPUT_DIR


def plot_history(history, do_cv=False, name="/model_loss"):
    keys = list(history.history.keys())
    plt.figure()
    plt.plot(history.history[keys[0]], label="train")
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if do_cv:
        plt.plot(history.history[keys[1]], label="test")
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
    plt.legend()
    plt.savefig(OUTPUT_DIR + name + ".png")
    plt.close()


def plot_model(model):
    with open(OUTPUT_DIR + '/model.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
