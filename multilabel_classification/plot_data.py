from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px

from multilabel_classification.multi_label_ensemble_classifier import MLEnsembleClassifier


def plot_pca_interactive(data, labels, mode='Training'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca_viz = PCA(n_components=3)
    principal_df = pd.DataFrame(pca_viz.fit_transform(data), columns=['PC1', 'PC2', 'PC3'])
    final_df = pd.concat([principal_df, labels], axis=1)
    final_df.rename(columns={"label": "Label"}, inplace=True)
    fig = px.scatter_3d(final_df, x='PC1', y='PC2', z='PC3',
                        color="Label",
                        title=mode + ' Data - PCA',
                        color_discrete_sequence=["red", "green", "lightcoral", "lime", "black", "cyan", "#feafda",
                                                  "blue", "magenta", "honeydew", "gray", "goldenrod", "darkmagenta",
                                                  "yellow",
                                                  "CornflowerBlue", "LightPink"])  # ,
    # symbol="label")
    fig.update_traces(marker=dict(size=9,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    fig.show()


def determine_labels(labels):
    labels_original = labels
    labels = str(labels)
    if 'White Identitarian' in labels:
        return 'White Identitarian'
    elif 'MRA' in labels:
        return 'MRA'
    elif 'Conspiracy' in labels:
        return 'Conspiracy'
    elif 'Libertarian' in labels:
        return 'Libertarian'
    elif 'AntiSJW' in labels:
        return 'AntiSJW'
    elif 'Socialist' in labels:
        return 'Socialist'
    elif 'ReligiousConservative' in labels:
        return 'ReligiousConservative'
    elif 'SocialJustice' in labels:
        return 'SocialJustice'
    elif 'MainstreamNews' in labels or 'MissingLinkMedia' in labels:
        return 'MainstreamNews'
    elif 'PartisanLeft' in labels:
        return 'PartisanLeft'
    elif 'PartisanRight' in labels:
        return 'PartisanRight'
    elif 'AntiTheist' in labels:
        return 'AntiTheist'
    elif len(labels_original) == 1:
        return labels_original[0].strip('__label__')
    else:
        return 'Other'


if __name__ == "__main__":
    ensemble = MLEnsembleClassifier(training_fraction=1)
    ensemble.load_data(serialized=True)
    ensemble.fit(serialized=True)
    training_data = ensemble.estimate_probabilities(mode='Training')
    training_y = training_data['label']
    training_x = training_data.drop(columns=['id',
                                             'title',
                                             'label',
                                             'comments',
                                             'snippets'])
    training_y = training_y.apply(determine_labels)
    plot_pca_interactive(training_x, training_y)

    testing_data = ensemble.estimate_probabilities(mode='Testing')
    testing_y = testing_data['label']
    testing_x = testing_data.drop(columns=['id',
                                                        'title',
                                                        'label',
                                                        'comments',
                                                        'snippets'])
    testing_y = testing_y.apply(determine_labels)
    # plot_tsne_interactive(testing_x, testing_y, mode='Testing')
    plot_pca_interactive(testing_x, testing_y, mode='Testing')
    print(ensemble.evaluate(roc=False))
