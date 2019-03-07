from interpret_layers import plot, print_kmeans, load_representations, get_cut_index, get_question_range, \
    parse_token_file, parse_prediction_index_range, get_support_range
import os
import argparse
from hotpot_sample_dict import samples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="sup_only", type=str, help="Name of task")
    parser.add_argument("--sample_name", default="2_boyband_formed_by", type=str, help="Sample name")
    parser.add_argument('--no_embedding', action='store_true', help="Whether to plot only the embedding")
    parser.add_argument('--no_layers', action='store_true', help="Whether to plot only the layers")
    parser.add_argument('--plot_3d', action='store_true', help="Whether to plot in 3d")

    return parser.parse_args()


def main():
    args = parse_args()

    sample_info = samples[args.sample_name] if args.sample_name in samples else {}

    path = os.path.join("hotpot_samples", args.sample_name, args.task_name)

    tokens_file = os.path.join(path, "tokens.txt")
    predictions_file = os.path.join(path, "nbest_predictions.json")

    tokens = parse_token_file(tokens_file)
    pred_index = parse_prediction_index_range(predictions_file)

    sample_info["tokens"] = tokens
    sample_info["pred_index"] = pred_index
    sample_info["sp_parts"] = sample_info["sp_parts"] if "sp_parts" in sample_info else [
        get_support_range(tokens)]
    sample_info["question_parts"] = sample_info["question_parts"] if "question_parts" in sample_info else [
        get_question_range(tokens)]

    print(sample_info["sp_parts"])

    inputs, embedding, layers = load_representations(path)

    cut_index = get_cut_index(inputs, tokens)

    if not args.no_embedding:
        title = "embedding"
        plot(sample_info, embedding[:cut_index], "pca", title, path, args.plot_3d)
        print_kmeans(sample_info, embedding[:cut_index], title)

    if not args.no_layers:
        for i, layer in enumerate(layers):
            title = "layer" + str(i)
            plot(sample_info, layer[0][:cut_index], "pca", title, path, args.plot_3d)
            print_kmeans(sample_info, layer[0][:cut_index], title)


if __name__ == "__main__":
    main()
