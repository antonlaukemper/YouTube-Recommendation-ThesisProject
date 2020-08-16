import pickle as pl
from matplotlib import pyplot as plt

from text_processing.experiments import vary_learning_rate, vary_grams
from text_processing.load_data import create_text_df, remove_missing_data
from text_processing.text_trainer import TextTrainer


def plot_epochs():
    with open('cross_validation_results/captions/epochs.pl', 'rb') as file:
        epochs = pl.load(file)
    with open('cross_validation_results/captions/cv_training.pl', 'rb') as file:
        training = pl.load(file)
    with open('cross_validation_results/captions/cv_testing.pl', 'rb') as file:
        validation = pl.load(file)

    fig, ax = plt.subplots()
    plt.plot(epochs, training, label="Training F1-Score")
    plt.plot(epochs, validation, label="Validation F1-Score")
    plt.legend(fontsize=15)
    ax.tick_params(length=6, width=2, labelsize= 15)
    ax.set_xlabel('Number of Epochs', fontsize=20)
    ax.set_ylabel("F1-Score", fontsize=20)
    plt.show()

def plot_lr():
    with open('cross_validation_results/captions/lr_epochs.pl', 'rb') as file:
        epochs = pl.load(file)
    with open('cross_validation_results/captions/lr_training.pl', 'rb') as file:
        training = pl.load(file)
    with open('cross_validation_results/captions/lr_testing.pl', 'rb') as file:
        validation = pl.load(file)

    fig, ax = plt.subplots()
    plt.plot(epochs, training, label="Training F1-Score" )
    plt.plot(epochs, validation, label="Validation F1-Score")
    plt.legend(fontsize=15)
    ax.tick_params(length=6, width=2, labelsize= 15)
    ax.set_xlabel('Learning Rate', fontsize=20)
    ax.set_ylabel("F1-Score", fontsize=20)
    plt.show()

def plot_ngrams():
    with open('cross_validation_results/captions/gr_epochs.pl', 'rb') as file:
        epochs = pl.load(file)
    with open('cross_validation_results/captions/gr_training.pl', 'rb') as file:
        training = pl.load(file)
    with open('cross_validation_results/captions/gr_testing.pl', 'rb') as file:
        validation = pl.load(file)

    fig, ax = plt.subplots()
    plt.plot(epochs, training, label="Training F1-Score" )
    plt.plot(epochs, validation, label="Validation F1-Score")
    plt.legend(fontsize=15)
    ax.tick_params(length=6, width=2, labelsize= 15)
    ax.set_xlabel('N-Grams', fontsize=20)
    ax.set_ylabel("F1-Score", fontsize=20)
    plt.show()

if __name__ == "__main__":
    plot_epochs()
    plot_lr()
    plot_ngrams()
    # caption_trainer = TextTrainer()
    # data = caption_trainer.get_data(training_fraction=1, serialized=False)[0]
    # data_for_training = remove_missing_data(data)
    # # vary_learning_rate(data_for_training, metric='f1-score')
    # vary_grams(data_for_training, metric='f1-score')