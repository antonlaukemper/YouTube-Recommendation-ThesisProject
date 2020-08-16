
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.express as px
import matplotlib.patches as mpatches
from sklearn.metrics import plot_roc_curve

from ensemble_classifier.ensemble_classifier import EnsembleClassifier


def plot(x, y, mode, features):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for index, row in x.iterrows():
        if features == 'Text':
            ax.scatter(row[0],
                       row[1],
                       row[2],
                       marker='o' if y.iloc[index] == '__label__political' else '^',
                       color='red' if y.iloc[index] == '__label__political' else 'green')
        else:
            ax.scatter(row[3],
                       row[4],
                       row[5],
                       marker='o' if y.iloc[index] == '__label__political' else '^',
                       color='red' if y.iloc[index] == '__label__political' else 'green')

    red_patch = mpatches.Patch(color='red', label='political')
    green_patch = mpatches.Patch(color='green', label='non-political')
    plt.legend(handles=[red_patch])
    ax.set_xlabel('Captions Prediction' if features == 'Text' else 'Related Channels Prediction')
    ax.set_ylabel('Comments Prediction' if features == 'Text' else 'Subscriptions Prediction')
    ax.set_zlabel('Snippets Prediction' if features == 'Text' else 'Cross-Channel Comments Prediction')
    ax.legend(handles=[red_patch, green_patch], bbox_to_anchor=(0,0), loc="lower left",
              bbox_transform=fig.transFigure, fontsize="medium")

    plt.show()


def plot_interactive(df, x, y, z, training=True):
    fig = px.scatter_3d(df, x=x, y=y, z=z,
                        hover_data=['id', 'title', 'related_channels_pred', 'subscriptions_pred',
                                    'cross_comments_pred'],
                        color="label",
                        title='Training Data' if training else 'Testing Data')
    fig.show()


def plot_roc(classifier, x, y):
    viz = plot_roc_curve(classifier, x, y)
    print(viz.roc_auc)
    plt.show()

if __name__ == "__main__":
    classifier = EnsembleClassifier(training_fraction=1)
    classifier.load_data()
    classifier.fit()
    training_x = classifier.ensemble_data_training
    training_y = classifier.ensemble_data_training