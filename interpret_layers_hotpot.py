from interpret_layers import plot, build_kmeans, load_representations, get_cut_index, get_question_range, \
    parse_token_file, parse_prediction_index_range, get_context_range, plot_for_demo
import os
import argparse
from hotpot_sample_dict import samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="context_384", type=str, help="Name of task")
    parser.add_argument("--sample_name", default="3_shirley_temple_gov_position", type=str, help="Sample name")
    parser.add_argument("--path", default="/Users/betty/Projekte/Datexis/Explainability/hotpot_samples", type=str,
                        help="Path to babi samples")
    parser.add_argument('--no_embedding', action='store_true', help="Whether to plot only the embedding")
    parser.add_argument('--no_layers', action='store_true', help="Whether to plot only the layers")
    parser.add_argument('--plot_3d', action='store_true', help="Whether to plot in 3d")
    parser.add_argument('--sentence_colored', action='store_true', help="Whether to color tokens by sentence")
    parser.add_argument('--k_means', action='store_true', help="Whether to write out k-means cluster")
    parser.add_argument('--paper_demo', action='store_true', help="Whether to plot for paper demo")

    return parser.parse_args()


def main():
    args = parse_args()

    sample_info = samples[args.sample_name] if args.sample_name in samples else {}

    path = os.path.join(args.path, args.sample_name, args.task_name)

    tokens_file = os.path.join(path, "tokens.txt")
    predictions_file = os.path.join(path, "nbest_predictions.json")

    tokens = parse_token_file(tokens_file)
    pred_range = parse_prediction_index_range(predictions_file)

    sample_info["tokens"] = tokens
    sample_info["pred_index"] = pred_range
    sample_info["sp_parts"] = sample_info["sp_parts"] if "sp_parts" in sample_info else [
        get_context_range(tokens)]
    sample_info["question_parts"] = sample_info["question_parts"] if "question_parts" in sample_info else [
        get_question_range(tokens)]

    print(sample_info["sp_parts"])

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
        layer_index = 2
        # highlighted_tokens = [10, 14, 29, 45]
        # highlighted_tokens = ["detention", "school"]
        highlighted_tokens = []
        sup_ranges = [range(20, 25), range(30, 39)]

        layer = layers[layer_index]

        title = f"Model fine-tuned on HotpotQA: Layer {layer_index}"
        plot_for_demo(sample_info, layer[0][:cut_index], title, path, highlighted_tokens, sup_ranges, pred_range)



if __name__ == "__main__":
    main()
