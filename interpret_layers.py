import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import numpy as np
from nltk.corpus import stopwords
import ast

from mpl_toolkits.mplot3d import Axes3D

punctuation = ['.', ',', '!', '?', '[SEP]', ')', '(', '"', "'", '-', '–']


def reduce(x, method, dims):
    if method == "pca":
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(x)

        return reduced.transpose()
    if method == "ica":
        ica = FastICA(n_components=dims, random_state=0)
        reduced = ica.fit_transform(x)

        return reduced.transpose()
    if method == "tsne":
        tsne = TSNE(n_components=dims)
        reduced = tsne.fit_transform(x)

        return reduced.transpose()


def euclidean(x):
    return euclidean_distances(x.numpy())


def cosine(x):
    return cosine_distances(x.numpy())


def get_color_for_token(i, sample, sentence_colored=False):

    if sentence_colored:
        num_colors = sample['tokens'].count('.') + 1
        cm = plt.get_cmap('gist_rainbow')
        slice = sample['tokens'][:i]
        token_sen = slice.count('.') + slice.count('[SEP]')
        return [cm(token_sen/num_colors)]
    
    sp_colors = ['darkgreen', 'greenyellow', 'limegreen']
    question_colors = ['cyan', 'blue', 'dodgerblue']

    prediction_ind = sample["pred_index"] if "pred_index" in sample else range(0, 0)
    sp_parts = sample["sp_parts"] if "sp_parts" in sample else [range(0, 0)]
    question_parts = sample["question_parts"] if "question_parts" in sample else [range(0, 0)]

    col = '0.5'

    for j, col_range in enumerate(sp_parts):
        if i in col_range:
            col = sp_colors[j]

    for k, col_range in enumerate(question_parts):
        if i in col_range:
            col = question_colors[k]

    if i in prediction_ind:
        col = 'red'

    return col


def save_and_close_plot(plot_path, title):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plt.savefig(os.path.join(plot_path, title) + ".png")

    # close plot
    plt.clf()


def plot(sample, x, reduce_method, title, path, plot_3d, sentence_colored=False):
    if plot_3d:
        plot3d(sample, x, reduce_method, title, path, sentence_colored)
    else:
        plot2d(sample, x, reduce_method, title, path, sentence_colored)


def plot2d(sample, x, reduce_method, title, path, sentence_colored=False):
    tokens = sample["tokens"]

    coords = reduce(x, reduce_method, 2)

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]

        col = get_color_for_token(i, sample, sentence_colored)

        plt.scatter(x, y, c=col)

        if i < len(tokens):
            plt.text(x + 0.1, y + 0.1, tokens[i], fontsize=6)
        else:
            plt.text(x + 0.1, y + 0.1, "PAD", fontsize=6)

    plt.title(title)
    save_and_close_plot(os.path.join(path, 'colored_plots'), title)


def plot3d(sample, x, reduce_method, title, path, sentence_colored=False):
    tokens = sample["tokens"]

    coords = reduce(x, reduce_method, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]
        z = coords[2][i]

        col = get_color_for_token(i, sample, sentence_colored)

        # plt.scatter(x, y, c=col)
        ax.scatter([x], [y], [z], c=col)

        if i < len(tokens):
            ax.text(x + 0.1, y + 0.1, z + 0.1, tokens[i], fontsize=6)
        else:
            ax.text(x + 0.1, y + 0.1, z + 0.1, "PAD", fontsize=6)

    ax.set_title(title)
    save_and_close_plot(os.path.join(path, 'colored_plots', '3D'), title)


def plot_kmeans(sample, x, reduce_method, title, path):
    colors = ['yellow', 'red', 'gray', 'orange', 'pink', 'blue', 'purple', 'green', 'cyan']
    tokens = sample["tokens"]

    coords = reduce(x, reduce_method, 2)

    kmeans = KMeans(n_clusters=9)
    kmeans.fit(x)

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]

        plt.scatter(x, y, c=colors[kmeans.labels_[i]])

        if i < len(tokens):
            plt.text(x + 0.001, y + 0.001, tokens[i], fontsize=6)
        else:
            plt.text(x + 0.001, y + 0.001, "PAD", fontsize=6)

    plt.title(title)
    save_and_close_plot(os.path.join(path, 'kmeans'), title)


def build_kmeans(sample, x, title):
    tokens = sample["tokens"]

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(x)

    clusters = {}

    summary = ""

    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        word = tokens[i]

        if 0 < i < tokens.index("[SEP]"):
            word = "*{}*".format(word)
        clusters[label].append(word)

    summary += title + "\n"

    for cluster in clusters.values():
        summary += str(cluster) + "\n"

    return summary


def plot_parts(sample, x, reduce_method, title, path):
    x = x[:, 50:100]
    plot(sample, x, reduce_method, title, os.path.join(path, '50-100'))


def track_changes(tokens, layers, cut_index, cut_index_question):
    stopWords = set(stopwords.words('english'))

    distance_diffs = []

    for l in range(len(layers) - 1):
        first = layers[l][0][:cut_index]
        second = layers[l + 1][0][:cut_index]

        distances1 = euclidean(first)[:cut_index_question]
        distances2 = euclidean(second)[:cut_index_question]
        diff = distances2 - distances1

        distance_diffs.append(diff)

    for i in range(cut_index_question):

        info = str(tokens[i]) + " <-"

        for j in range(len(layers) - 1):

            diff = distance_diffs[j][i]

            k_min = np.argsort(diff)

            info += " ("

            cnt = 0
            for k in k_min:

                diff_token = tokens[k]

                if diff_token.lower() not in stopWords and diff_token not in punctuation:
                    distance_loss = str(diff[k])
                    info += str(diff_token)

                    if cnt > 2:
                        info += ")"
                        break
                    else:
                        info += ", "
                        cnt += 1
        print(info)


def track_neighbors(tokens, pred_ind, layers, cut_index, cut_index_question):
    # stopWords = set(stopwords.words('english'))

    for l in range(len(layers)):
        print("-------- LAYER " + str(l) + " --------")
        vectors = layers[l][0][:cut_index]

        nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(vectors)
        distances, indices = nbrs.kneighbors(vectors)

        q_neighbors = indices[:cut_index_question]
        pred_neighbors = indices[pred_ind[0]:pred_ind[-1] + 1]
        for n in pred_neighbors:
            print_neighbors(tokens, n)

        for m in q_neighbors:
            print_neighbors(tokens, m)


def print_neighbors(tokens, neighbors):
    token_i = neighbors[0]
    info = str(tokens[token_i]) + ": ("
    for i in range(1, len(neighbors)):
        info += str(tokens[neighbors[i]]) + ", "
    info += ")"
    print(info)


def load_representations(path):
    inputs_file = os.path.join(path, "inputs.pkl")
    embedding_file = os.path.join(path, "embedding.pkl")
    layers_file = os.path.join(path, "encoded_layers.pkl")

    inputs = torch.load(inputs_file, map_location='cpu')
    embedding = torch.load(embedding_file, map_location='cpu')[0]
    layers = torch.load(layers_file, map_location='cpu')

    return inputs, embedding, layers


def load_old_representations(path):
    inputs_file = os.path.join(path, "inputs.pkl")
    embedding_file = os.path.join(path, "embedding.pkl")
    layers_file = os.path.join(path, "encoded_layers.pkl")

    inputs = torch.load(inputs_file, map_location='cpu')[0]
    embedding = torch.load(embedding_file, map_location='cpu')[0]
    layers = torch.load(layers_file, map_location='cpu')[0]

    return inputs, embedding, layers


def get_cut_index(inputs, tokens):
    if len((inputs == 0).nonzero()) != 0 and len(tokens) != 384:
        cut_index = (inputs == 0).nonzero()[0][1].item()
    else:
        cut_index = len(tokens)

    print(cut_index)
    print(len(tokens))

    return cut_index


def get_question_range(tokens):
    return range(1, tokens.index("[SEP]"))


def get_support_range(tokens):
    return range(tokens.index("[SEP]"), len(tokens) - 1)


def parse_token_file(path):
    with open(path, encoding='utf-8') as f:
        tokens = ast.literal_eval(f.read())
        return tokens


def parse_prediction_index_range(path):
    with open(path, encoding='utf-8') as f:
        preds = ast.literal_eval(f.read())

        for entry in preds.values():
            r = range(int(entry[0]["start"]), int(entry[0]["end"] + 1))
            return r
