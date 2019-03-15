from interpret_layers import plot, load_representations, build_kmeans, get_cut_index, get_question_range, \
    parse_token_file, parse_prediction_index_range
import os
import argparse

from babi_sample_dict import samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--babi_task", default="qa13_base_cl", type=str, help="BABI Task name")
    parser.add_argument("--sample_name", default="13cl_where_is_sandra", type=str, help="Sample name")
    parser.add_argument('--no_embedding', action='store_true', help="Whether to plot only the embedding")
    parser.add_argument('--no_layers', action='store_true', help="Whether to plot only the layers")
    parser.add_argument('--plot_3d', action='store_true', help="Whether to plot in 3d")
    parser.add_argument('--sentence_colored', action='store_true', help="Whether to color tokens by sentence")

    return parser.parse_args()


def main():
    args = parse_args()

    path = os.path.join("babi_samples", args.babi_task, args.sample_name)

    tokens_file = os.path.join(path, "tokens.txt")
    predictions_file = os.path.join(path, "nbest_predictions.json")

    sample_info = samples[args.sample_name] if args.sample_name in samples else {}

    tokens = parse_token_file(tokens_file)
    question_parts = get_question_range(tokens)

    if os.path.exists(predictions_file):
        pred_index = parse_prediction_index_range(predictions_file)
        sample_info["pred_index"] = pred_index

    sample_info["tokens"] = tokens
    sample_info["question_parts"] = [question_parts]

    inputs, embedding, layers = load_representations(path)

    cut_index = get_cut_index(inputs, tokens)

    kmeans_summary = ""

    if not args.no_embedding:
        title = "embedding"
        plot(sample_info, embedding[:cut_index], "pca", title, path, args.plot_3d, args.sentence_colored)
        kmeans_summary += build_kmeans(sample_info, embedding[:cut_index], title)

    if not args.no_layers:
        for i, layer in enumerate(layers):
            title = "layer" + str(i)
            plot(sample_info, layer[0][:cut_index], "pca", title, path, args.plot_3d, args.sentence_colored)
            kmeans_summary += build_kmeans(sample_info, layer[0][:cut_index], title)

    with open(os.path.join(path, "kmeans_clusters.txt"), "w") as cluster_file:
        cluster_file.write(kmeans_summary)


if __name__ == "__main__":
    main()
