import os
import logging
import argparse
import glob
import numpy as np
import pandas as pd
import shutil
from matplotlib import pyplot

from sklearn_models import evaluate_sklearn_model
from gpt3_models import evaluate_gpt3_model, classify_message


logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_TEXT_DATA_LEN = 200
MAX_SKLEARN_ML_FEATURES = 1000


def generate_datasets(
    path_output_folder="spam_data",
    path_input_tsv_file="spam_data/SMSSpamCollection",
    column_text="text",
    column_label="label",
    positive_label="Spam",
    train_sample_size_list=[1024, 512, 32, 8, 2],
    test_sample_size=256,
    max_text_data_len=MAX_TEXT_DATA_LEN
):
    """
    generate train and test datasets from path_input_tsv_file.
    :param path_output_folder: output folder
    :param path_input_tsv_file: input data file
    :param column_text: column for text data
    :param column_label: column for label data
    :param positive_label: label value for positive samples
    :param train_sample_size_list: a list of training sample sizes
    :param test_sample_size: test sample size
    :param max_text_data_len: max text length for pre-processing
    """
    # load the input tsv file
    df = pd.read_csv(path_input_tsv_file, header=None,
                     sep="\t", names=[column_label, column_text])

    # pre-process the label and text columns.
    df["label"] = df["label"].apply(lambda x: x.capitalize())
    df["text"] = df["text"].apply(lambda x: x[:max_text_data_len])
    logger.info("{}, df.shape:{}".format(path_input_tsv_file, df.shape))
    logger.info("df['label'].value_counts:{}".format(
        df["label"].value_counts()))

    # suffle the df and split it into train and test datasets.
    df = df.sample(frac=1, replace=False)
    df_test, df_train = df[:test_sample_size], df[test_sample_size:]
    logger.info("df_test.shape:{}".format(df_test.shape))
    logger.info("df_test['label'].value_counts:{}".format(
        df_test["label"].value_counts()))

    logger.info("df_train.shape:{}".format(df_train.shape))
    logger.info("df_train['label'].value_counts:{}".format(
        df_train["label"].value_counts()))

    path_output = os.path.join(path_output_folder, "train.csv")
    df_train.to_csv(path_output, sep="\t", index=False)

    path_output = os.path.join(path_output_folder, "test.csv")
    df_test.to_csv(path_output, sep="\t", index=False)

    is_positive = df_train["label"] == positive_label
    df_train_positive = df_train[is_positive]
    df_train_negative = df_train[~is_positive]

    # generate training datasets
    for train_sample_size in train_sample_size_list:
        positive_sample_size = negative_sample_size = train_sample_size//2
        df_train_positive = df_train_positive.sample(
            positive_sample_size, replace=False)
        df_train_negative = df_train_negative.sample(
            negative_sample_size, replace=False)
        # create a balanced dataset both from positive and negative samples
        df_train_all = pd.concat(
            [df_train_positive, df_train_negative]).sample(frac=1, replace=False)

        path_output = os.path.join(
            path_output_folder, "train_{}.csv".format(train_sample_size))
        df_train_all.to_csv(path_output, sep="\t", index=False)
        logger.info("{}, df_train_all.shape:{}".format(
            path_output, df_train_all.shape))
        logger.info("df_train_all['label'].value_counts:{}".format(
            df_train_all["label"].value_counts()))


def evaluate_model_with_train_sample_sizes(
    path_results_output,
    model_name="RandomForest",
    path_data_folder="spam_data",
    train_sample_size_list=[2, 8, 32, 512, 1024],
    data_type="Message",
    column_text="text",
    column_label="label",
    positive_label="Spam",
    negative_label="Ham",
    fine_tune=True,
    fine_tuned_model="",
    fine_tune_context_sample_size=4,
    prompt_context_sample_size=3,
    sleep_time=1
):
    """
    evaluate a ML model with train sample sizes.
    :param path_results_output: path for outputs
    :param model_name: model name
    :param path_data_folder: path for data
    :param train_sample_size_list: a list of train sample sizes
    :param data_type: data type
    :param column_text: the column for text data
    :param column_label: the column for label data
    :param positive_label: the positive label value
    :param negative_label: the negative label value
    :param fine_tune: True to fine_tune
    :param fine_tuned_model: fine_tuned model name
    :param fine_tune_context_sample_size: the sample size for fine-tunning prompt data
    :param prompt_context_sample_size=3: the sample size for context prompt
    :param sleep_time: sleep time in seconds
    """
    logger.info("evaluate_model_with_train_sample_sizes:{}".format(locals()))

    f1_score_list = []
    for train_sample_size in train_sample_size_list:
        if train_sample_size > 0:
            path_train_data = os.path.join(
                path_data_folder, "train_{}.csv".format(train_sample_size))
        else:
            path_train_data = os.path.join(path_data_folder, "train.csv")
        path_test_data = os.path.join(path_data_folder, "test.csv")

        if model_name in ["RandomForest", "LogisticRegression"]:
            report_test = evaluate_sklearn_model(
                path_train_data=path_train_data,
                path_test_data=path_test_data,
                model_name=model_name,
                max_features=MAX_SKLEARN_ML_FEATURES,
                column_text=column_text,
                column_label=column_label,
                positive_label=positive_label
            )
        else:
            report_test = evaluate_gpt3_model(
                path_train_data,
                path_test_data=path_test_data,
                model_name=model_name,
                data_type=data_type,
                column_text=column_text,
                column_label=column_label,
                positive_label=positive_label,
                negative_label=negative_label,
                fine_tune=fine_tune,
                fine_tuned_model=fine_tuned_model,
                fine_tune_context_sample_size=fine_tune_context_sample_size,
                prompt_context_sample_size=prompt_context_sample_size,
                sleep_time=sleep_time
            )
            
        if report_test is not None:
            test_f1_score = report_test["weighted avg"]["f1-score"]
            report_item = {"train_size": train_sample_size}
            report_item.update(
                {"test_"+k: v for k, v in report_test["weighted avg"].items()})
            f1_score_list.append(report_item)

            logger.info("model_name:{}, train_size:{}, test_f1_score:{}".format(
                model_name, train_sample_size, test_f1_score))

    #store the f1 scores
    df = pd.DataFrame(f1_score_list)
    df.to_csv(path_results_output, index=False)


def plot_f1_results(
    plot_item_list=[("RandomForest", "orange", "o" ,"../**/results_randomforest.csv")], 
    path_output="result_f1_plot.pdf"
):
    """
    plot f1 results.
    :param plot_item_list: a list of plot data items
    :param path_output: file path for output image 
    """
    pyplot.clf()

    for label, color, marker, path_pattern in plot_item_list:
        f1_list=[]
        for path_file in glob.glob(path_pattern, recursive=1):
            df = pd.read_csv(path_file)
            f1_list.append(df["test_f1-score"])
        if len(f1_list) == 0:
            continue
        x = [str(size) for size in df["train_size"]]

        f1_list = np.array(f1_list)
        #the mean f1 values
        f1_mean = np.mean(f1_list, axis=0)
        #the std f1 values
        f1_std = np.std(f1_list, axis=0)
        f1_upper = f1_mean + f1_std
        f1_lower = f1_mean - f1_std

        pyplot.plot(x, f1_mean, marker=marker, color=color, label=label)
        pyplot.fill_between(x, f1_lower, f1_upper, color=color, alpha=.2)

    pyplot.grid()
    pyplot.ylabel("F1-score")
    pyplot.xlabel("Training sample size")
    pyplot.legend(loc="lower right")
    pyplot.savefig(path_output)


def run_experiments(path_data_folder="spam_data"):
    """
    run experiments with data sets in the data folder.
    :param path_data_folder: folder for the data sets
    """
    # generate train and test datasets.
    generate_datasets(
        path_output_folder=path_data_folder,
        path_input_tsv_file=os.path.join(
            path_data_folder, "SMSSpamCollection"),
        column_text="text",
        column_label="label",
        positive_label="Spam",
        #the sample size list for training datasets should be sorted in descending order.
        train_sample_size_list=[1024, 512, 32, 8, 2],
        test_sample_size=256
    )

    # evaluate RandomForest model
    path_results_randomforest = os.path.join(
        path_data_folder, "results_randomforest.csv")
    evaluate_model_with_train_sample_sizes(
        path_results_output=path_results_randomforest,
        model_name="RandomForest",
        path_data_folder=path_data_folder,
        #the sample size list for evaluation should be sorted in ascending order.
        train_sample_size_list=[2, 8, 32, 512, 1024]
    )

    # evaluate LogisticRegression model
    path_results_logisticregression = os.path.join(
        path_data_folder, "results_logisticregression.csv")
    evaluate_model_with_train_sample_sizes(
        path_results_output=path_results_logisticregression,
        model_name="LogisticRegression",
        path_data_folder=path_data_folder,
        train_sample_size_list=[2, 8, 32, 512, 1024]
    )

    # evaluate GPT-3 few-shot model with few samples of [2, 8, 32]
    path_results_gpt3_fewshot = os.path.join(
        path_data_folder, "results_gpt3_fewshot.csv")
    evaluate_model_with_train_sample_sizes(
        path_results_output=path_results_gpt3_fewshot,
        model_name="text-davinci-002",
        path_data_folder=path_data_folder,
        train_sample_size_list=[2, 8, 32],
        fine_tune=False
    )

    # # evaluate GPT-3 fine-tuning model with a few samples of [512, 1024]
    path_results_gpt3_finetune = os.path.join(
        path_data_folder, "results_gpt3_finetune.csv")
    evaluate_model_with_train_sample_sizes(
        path_results_output=path_results_gpt3_finetune,
        model_name="davinci",
        path_data_folder=path_data_folder,
        train_sample_size_list=[512, 1024],
        fine_tune=True
    )

    #plot f1 results
    plot_items = [
        #(label, color, mark, path_pattern)
        ("GPT3_Fewshot", "blue", "*", path_results_gpt3_fewshot),
        ("GPT3_Finetune", "red", "*", path_results_gpt3_finetune),
        ("RandomForest", "orange", "o", path_results_randomforest),
        ("LogisticRegression", "green", "v", path_results_logisticregression)
    ]
    plot_f1_results(plot_items, path_output=os.path.join(path_data_folder, "results_f1_plot.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam detection")
    parser.add_argument(
        "--run_type",
        help="classify_message or evaluate_approaches",
        default="classify_message"
    )
    parser.add_argument(
        "--message",
        help="message to be classified",
        default=""
    )
    parser.add_argument(
        "--path_train_data",
        help="file path for training samples",
        default="spam_data/experiment_1/train_2.csv"
    )

    parser.add_argument(
        "--path_data_folder",
        help="file folder for data",
        default="spam_data"
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        help="number of experiments",
        default=5
    )
    args = parser.parse_args()

    if args.run_type == "classify_message":
        classify_message(
            message=args.message,
            path_train_data=args.path_train_data
        )
    else:
        #run experiments
        for ix in range(1, args.num_experiments+1):
            path_data_folder = os.path.join(args.path_data_folder, "experiment_{}".format(ix))
            logger.info("==== experiment_{}, path_data_folder:{}".format(ix, path_data_folder))
            os.makedirs(path_data_folder, exist_ok=True)
            path_src_data = os.path.join(args.path_data_folder, "SMSSpamCollection")
            path_dest_data = os.path.join(path_data_folder, "SMSSpamCollection")
            shutil.copyfile(path_src_data, path_dest_data)
            run_experiments(path_data_folder=path_data_folder)
        
        #plot mean f1 with all experiment results.
        plot_items = [
            #(label, color, marker, path_pattern)
            ("GPT3_Fewshot", "blue", "*", os.path.join(args.path_data_folder, "**", "results_gpt3_fewshot.csv")),
            ("GPT3_Finetune", "red", "*", os.path.join(args.path_data_folder, "**", "results_gpt3_finetune.csv")),
            ("RandomForest", "orange", "o", os.path.join(args.path_data_folder, "**", "results_randomforest.csv")),
            ("LogisticRegression", "green", "v", os.path.join(args.path_data_folder, "**", "results_logisticregression.csv"))
        ]
        plot_f1_results(plot_items, path_output=os.path.join(args.path_data_folder, "results_mean_f1_plot.pdf"))
