from interpret_layers import plot, load_representations, build_kmeans, get_cut_index, get_question_range, \
    parse_token_file, parse_prediction_index_range, plot_for_demo
import os
import argparse

from babi_sample_dict import samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_name", default="qa15_3", type=str, help="Sample name")
    parser.add_argument("--path", default="/Users/betty/Projekte/Datexis/Explainability/babi_samples/bert_babi_plots/base_colored", type=str, help="Path to babi samples")
    parser.add_argument('--no_embedding', action='store_true', help="Whether to plot only the embedding")
    parser.add_argument('--no_layers', action='store_true', help="Whether to plot only the layers")
    parser.add_argument('--plot_3d', action='store_true', help="Whether to plot in 3d")
    parser.add_argument('--sentence_colored', action='store_true', help="Whether to color tokens by sentence")
    parser.add_argument('--k_means', action='store_true', help="Whether to write out k-means cluster")
    parser.add_argument('--paper_demo', action='store_true', help="Whether to plot for paper demo")

    return parser.parse_args()


def main():
    args = parse_args()

    path = os.path.join(args.path, args.sample_name)

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

        if args.k_means:
            kmeans_summary += build_kmeans(sample_info, embedding[:cut_index], title)

    if not args.no_layers:
        for i, layer in enumerate(layers):
            title = "layer" + str(i)
            plot(sample_info, layer[0][:cut_index], "pca", title, path, args.plot_3d, args.sentence_colored)

            if args.k_means:
                kmeans_summary += build_kmeans(sample_info, layer[0][:cut_index], title)

    if args.k_means:
        with open(os.path.join(path, "kmeans_clusters.txt"), "w") as cluster_file:
            cluster_file.write(kmeans_summary)

    if args.paper_demo:
        layer_index = 5
        # highlighted_tokens = [10, 14, 29, 45]
        # highlighted_tokens = ["cat", "sheep"]
        highlighted_tokens = []
        sup_ranges = [range(20, 25), range(30, 39)]
        pred_range = range(13, 14)

        layer = layers[layer_index]

        title = f"Model fine-tuned on bAbI: Layer {layer_index}"
        plot_for_demo(sample_info, layer[0][:cut_index], title, path, highlighted_tokens, sup_ranges, pred_range)


if __name__ == "__main__":
    main()
