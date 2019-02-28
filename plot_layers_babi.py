import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import ast
import argparse

from babi_sample_dict import samples

from mpl_toolkits.mplot3d import Axes3D


def plot(sample, x, title, path):
    tokens = sample["tokens"]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(x)

    coords = reduced.transpose()

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]

        col = get_color_for_token(i, sample)

        plt.scatter(x, y, c=col)

        if i < len(tokens):
            plt.text(x + 0.1, y + 0.1, tokens[i], fontsize=6)
        else:
            plt.text(x + 0.1, y + 0.1, "PAD", fontsize=6)

    plt.title(title)
    save_and_close_plot(os.path.join(path, 'colored_plots'), title)


def plot(sample, x, title, path):
    tokens = sample["tokens"]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(x)

    coords = reduced.transpose()

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]

        col = get_color_for_token(i, sample)

        plt.scatter(x, y, c=col)

        if i < len(tokens):
            plt.text(x + 0.1, y + 0.1, tokens[i], fontsize=6)
        else:
            plt.text(x + 0.1, y + 0.1, "PAD", fontsize=6)

    plt.title(title)
    save_and_close_plot(os.path.join(path, 'colored_plots'), title)


def plot3d(sample, x, title, path):
    tokens = sample["tokens"]

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(x)

    coords = reduced.transpose()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, val in enumerate(coords[0]):
        x = coords[0][i]
        y = coords[1][i]
        z = coords[2][i]

        col = get_color_for_token(i, sample)

        # plt.scatter(x, y, c=col)
        ax.scatter([x], [y], [z], c=col)

        if i < len(tokens):
            ax.text(x + 0.1, y + 0.1, z + 0.1, tokens[i], fontsize=6)
        else:
            ax.text(x + 0.1, y + 0.1, z + 0.1, "PAD", fontsize=6)

    ax.set_title(title)
    save_and_close_plot(os.path.join(path, 'colored_plots', '3D'), title)


def save_and_close_plot(plot_path, title):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plt.savefig(os.path.join(plot_path, title) + ".png")

    # close plot
    plt.clf()


def get_color_for_token(i, sample):
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


def get_question_range(tokens):
    return range(1, tokens.index("[SEP]"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--babi_task", default="qa11_base", type=str, help="BABI Task name")
    parser.add_argument("--sample_name", default="correct_where_is_sandra", type=str, help="BABI Task name")
    parser.add_argument('--no_embedding', action='store_true', help="Whether to plot only the embedding")
    parser.add_argument('--no_layers', action='store_true', help="Whether to plot only the layers")
    parser.add_argument('--plot_3d', action='store_true', help="Whether to plot in 3d")

    args = parser.parse_args()

    path = os.path.join("babi_samples", args.babi_task, args.sample_name)

    tokens_file = os.path.join(path, "tokens.txt")
    inputs_file = os.path.join(path, "inputs.pkl")
    embedding_file = os.path.join(path, "embedding.pkl")
    layers_file = os.path.join(path, "encoded_layers.pkl")
    predictions_file = os.path.join(path, "nbest_predictions.json")

    sample_info = samples[args.sample_name] if args.sample_name in samples else {}

    tokens = parse_token_file(tokens_file)
    pred_index = parse_prediction_index_range(predictions_file)
    question_parts = get_question_range(tokens)

    sample_info["tokens"] = tokens
    sample_info["pred_index"] = pred_index
    sample_info["question_parts"] = [question_parts]

    inputs = torch.load(inputs_file, map_location='cpu')
    embedding = torch.load(embedding_file, map_location='cpu')[0]
    layers = torch.load(layers_file, map_location='cpu')

    if len((inputs == 0).nonzero()) != 0 and len(tokens) != 384:
        cut_index = (inputs == 0).nonzero()[0][1].item()
    else:
        cut_index = len(tokens)

    # check if cut index corresponds with loaded tokens
    print(cut_index)
    print(len(tokens))

    if not args.no_embedding:
        title = "embedding"
        if args.plot_3d:
            plot3d(sample_info, embedding[:cut_index], title, path)
        else:
            plot(sample_info, embedding[:cut_index], "embedding", path)

    if not args.no_layers:
        for i, layer in enumerate(layers):
            title = "layer" + str(i)
            if args.plot_3d:
                plot3d(sample_info, layer[0][:cut_index], title, path)
            else:
                plot(sample_info, layer[0][:cut_index], title, path)


if __name__ == "__main__":
    main()
