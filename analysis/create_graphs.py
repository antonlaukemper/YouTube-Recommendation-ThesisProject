import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.drawing.layout import *
import plotly.graph_objects as go
import plotly.express as px
from utils.data_IO import read_channels


def plot_channels(path_P, path_NP, path_U, merge=False, content='related_channels', only_known=False, directed=False,
                  output='network_plot.html', line_width=1.0):
    channels = read_channels(path_P)
    if merge:
        channels_NP = read_channels(path_NP)
        channels.update(channels_NP)
        channels_U = read_channels(path_U)
        # channels.update(channels_U)
    network_data = pd.DataFrame({'source': [],
                                 'target': []})
    for channel, value in tqdm(channels.items()):
        for affiliate in value[content]:
            tmp = pd.DataFrame({'source': [channel],
                                'target': [affiliate]})
            network_data = network_data.append(tmp)

    network_data['index'] = range(0, len(network_data))
    network_data.set_index('index', inplace=True)

    # get basic information about unknown channels, but only if in related_channels mode, otherwise there are too many
    if content == 'related_channels':
        # get channel info via API
        # unknown_channels = []
        # for index, row in network_data.iterrows():
        #     if row['target'] not in channels.keys():
        #         unknown_channels.append(row['target'])
        # this method uses api quota so it should be called rarely, instead, the file should be read from disk
        # related_channel_info = get_channel_snippets(unknown_channels)

        # or use previous execution of that function, that was written to a file
        related_channel_info = read_channels('related_channels.json')

    if only_known:
        network_data['target'] = network_data['target'].apply(lambda x: 'unknown' if x not in channels.keys() else x)
        network_data = network_data[network_data['target'] != 'unknown']
        # for index, row in tqdm(network_data.iterrows()):
        #     if row['target'] not in channels.keys():
        #         network_data.drop(index, inplace=True)

    di_G = nx.from_pandas_edgelist(network_data, create_using=nx.DiGraph())
    G = nx.from_pandas_edgelist(network_data)
    coordinates = spring_layout(G, seed=42, iterations=50)  # , iterations=1 ,k=1/sqrt(n) = 0.03
    # coordinates = kamada_kawai_layout(G)  # , iterations=1 ,k=1/sqrt(n)
    # coordinates = spectral_layout(G)
    edge_x = []
    edge_y = []
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    for edge in di_G.edges():
        x0, y0 = coordinates[edge[0]]
        x1, y1 = coordinates[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        x_0.append(x0)
        y_0.append(y0)
        x_1.append(x1)
        y_1.append(y1)

    node_df = pd.DataFrame({'ChannelId': [],
                            'Title': [],
                            'Subscribers': [],
                            'ChannelViews': [],
                            '#related_channels': [],
                            '#subscriptions': [],
                            'in_degree': [],
                            'x': [],
                            'y': [],
                            'label': [],
                            'labels': [],
                            'known': []})

    for node in G.nodes():
        x, y = coordinates[node]
        tmp = pd.DataFrame({'ChannelId': [node],
                            'Title': [channels[node]['ChannelTitle']] if node in channels.keys() else
                            [related_channel_info[node]['ChannelTitle']] if node in related_channel_info.keys() else [
                                'Unknown'],
                            'Subscribers': [channels[node]['Subs']] if node in channels.keys() else
                            [related_channel_info[node]['Subs']] if node in related_channel_info.keys() else [
                                'Unknown'],
                            'ChannelViews': [channels[node]['ChannelViews']] if node in channels.keys() else
                            [related_channel_info[node]['ChannelViews']] if node in related_channel_info.keys() else [
                                'Unknown'],
                            '#related_channels': [len(channels[node]['related_channels'])] if node in channels.keys()
                            else [
                                related_channel_info[node][
                                    'related_channels']] if node in related_channel_info.keys() else [
                                0],
                            '#subscriptions': [len(channels[node]['subscriptions'])] if node in channels.keys() else [
                                0],
                            'in_degree': di_G.in_degree[node],
                            'x': [x],
                            'y': [y],
                            # 'label': [str(channels[node]['SoftTags'][0])] if node in channels.keys() else ['Unknown'],
                            'label': ['Unknown' if node not in channels.keys() else 'Non-Political'
                            if channels[node]['SoftTags'][0] == 'Non-Political' else 'Unlabeled'
                            if channels[node]['SoftTags'][0] == 'UNLABELED' else 'Political'],
                            'labels': [determine_labels(channels[node]['SoftTags'])] if node in channels.keys() else [
                                'Unknown'],
                            'known': str(node in channels.keys())})
        node_df = node_df.append(tmp)

    if directed:
        fig = px.scatter(node_df, x="x", y="y", color="labels",  # symbol='known',
                         hover_data=['Title', 'ChannelId', 'labels', '#subscriptions'], size='in_degree',
                         color_discrete_sequence=["lightcoral", "red", "green", "lime", "orange", "cyan",
                                                  "mediumslateblue",
                                                  "blue", "magenta", "honeydew", "gray", "goldenrod", "teal", "yellow",
                                                  "CornflowerBlue", "LightPink"]).update_layout(dict(
            annotations=[
                dict(ax=x_0[i], ay=y_0[i], axref='x', ayref='y',
                     x=x_1[i], y=y_1[i], xref='x', yref='y',
                     showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1) for i in range(0, len(x_0))
            ]
        ))
    else:
        fig = px.scatter(node_df, x="x", y="y", color="label",
                         hover_data=['Title', 'ChannelId', 'labels'], # size='in_degree',
                         color_discrete_sequence=["red", "green", "lightcoral", "lime", "black", "cyan", "#feafda",
                                                  "blue", "magenta", "honeydew", "gray", "goldenrod", "darkmagenta",
                                                  "yellow",
                                                  "CornflowerBlue", "LightPink"])

        fig2 = go.Figure(data=go.Scatter(x=edge_x, y=edge_y,
                                         line=dict(width=line_width, color='#888'),
                                         hoverinfo='skip',
                                         mode='lines',
                                         showlegend=False
                                         ))

        figTotal = go.Figure()
        figTotal.add_trace(fig2['data'][0])
        for trace in fig['data']:
            figTotal.add_trace(trace)
        # figTotal.add_trace(fig['data'][1])

        # fig.add_scatter(x=edge_x, y=edge_y,
        #                         line=dict(width=line_width, color='#888'),
        #                         hoverinfo='skip',
        #                         mode='lines',
        #                         showlegend=False
        #                         )

    figTotal.update_traces(marker=dict(size=10,
                                       line=dict(width=1,
                                                 color='DarkSlateGrey')),
                           selector=dict(mode='markers'))

    figTotal.update_xaxes(showgrid=False, zeroline=False, visible=False)
    figTotal.update_yaxes(showgrid=False, zeroline=False, visible=False)

    figTotal.show()
    figTotal.write_html(output)


def determine_labels(labels):
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
    elif 'Social Justice' in labels:
        return 'Social Justice'
    elif 'Mainstream News' in labels or 'Missing Link Media' in labels:
        return 'Mainstream News'
    elif 'PartisanLeft' in labels:
        return 'PartisanLeft'
    elif 'PartisanRight' in labels:
        return 'PartisanRight'
    elif 'AntiTheist' in labels:
        return 'AntiTheist'
    elif len(labels) == 1:
        return labels[0]
    else:
        return 'Other'


if __name__ == "__main__":
    plot_channels( '../output/all_merged.json',
        #'../output/final_data/P_channelData.json',
        '../output/final_data/NP_channelData.json',
        '../output/final_data/U_channelData_multilabel.json',
        merge=False,
        content='cross_comments',
        only_known=True, directed=False,
        output='results/cross_comments_binary.html',
        line_width=1.8)
    print('done')
