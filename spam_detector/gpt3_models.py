import os
import time
import pandas as pd
import logging
from sklearn.metrics import classification_report
import openai


# set your openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")
# disable openai's logging messages
logging.getLogger("openai").setLevel(logging.ERROR)

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = "Classify the {data_type} as {positive_label} or {negative_label}."
EXAMPLE_TEMPLATE = "\n{data_type}: {text}\nLabel: {label}"
QUERY_TEMPLATE = "\n{data_type}: {text}\nLabel:"


def generate_prompt_text(
    df,
    data_type="Message",
    positive_label="Spam",
    negative_label="Ham",
    column_text="text",
    column_label="label"
):
    """
    generate a prompt from the input df.
    :param df: data frame for input data
    :param data_type: the type of data
    :param positive_label: the value for positive label 
    :param negative_label: the value for negative label
    :param column_text: column for text
    :param column_label: column for label
    """
    prompt = PROMPT_TEMPLATE.format(
        data_type=data_type, positive_label=positive_label, negative_label=negative_label)
    if df is None or len(df) == 0:
        return prompt

    for _ix, row in df.iterrows():
        text = row[column_text]
        label = row[column_label]
        example_text = EXAMPLE_TEMPLATE.format(
            data_type=data_type, text=text, label=label)
        prompt += example_text
    return prompt


def get_openai_completion(
    prompt,
    model_name="text-davinci-002",
    max_tokens=6,
    sleep_time=60
):
    """
    get a completion response from openai.
    :param prompt: the input prompt
    :param model_name: model name
    :param max_tokens: the max token size for response
    :param sleep_time: sleep time in seconds
    """
    label = None
    while True:
        try:
            logging.debug("prompt:{}".format(prompt))
            res = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0,
                stop="\n"
            )
            logging.debug("res:{}".format(res))

            #remove the first white space and return the first word as a label.
            completion = res["choices"][0]["text"].strip()
            label = completion.split()[0]
            break
        except openai.error.RateLimitError as ex:
            logging.info("RateLimitError, ex:{}".format(ex))
            time.sleep(sleep_time)
        except openai.error.APIConnectionError as ex:
            logging.info("APIConnectionError, ex:{}".format(ex))
            time.sleep(sleep_time)
    return label


def upload_train_jsonl_file(
    path_jsonl,
    df_train,
    data_type="Message",
    positive_label="Spam",
    negative_label="Ham",
    column_text="text",
    column_label="label",
    purpose="fine-tune",
    fine_tune_context_sample_size=4
):
    """
    upload a train jsonl file to openai.
    :param path_jsonl: the path for training jsonl file
    :param df_train: df for training data
    :param data_type: data type
    :param positive_label: the value for positive label
    :param negative_label: the value for negative label
    :param column_text: column for text
    :param column_label: column for label
    :param purpose: the purpose of training data, fine-tune or classifications
    :param fine_tune_context_sample_size: the sampe size for prompt context
    """
    text_label = "Label:"
    dict_items = []
    for ix in range(0, len(df_train), fine_tune_context_sample_size):
        df_items = df_train.iloc[ix:ix+fine_tune_context_sample_size]
        context_prompt = generate_prompt_text(
            df_items, data_type=data_type,
            positive_label=positive_label, negative_label=negative_label,
            column_text=column_text, column_label=column_label)
        label_idx = context_prompt.rfind(text_label) + len(text_label)
        prompt = context_prompt[:label_idx]
        completion = context_prompt[label_idx:]
        dict_items.append({"prompt": prompt, "completion": completion})

    df = pd.DataFrame(dict_items)
    df.to_json(path_jsonl, orient="records", lines=True)
    logger.info("train_jsonl:{}, df.shape:{}".format(path_jsonl, df.shape))

    #update the file to openai
    try:
        res = openai.File.create(file=open(path_jsonl), purpose=purpose)
        train_file_id = res["id"]
    except Exception as ex:
        logger.error("openai.File.create() got an error, ex:{}".format(ex))
        train_file_id = None
    return train_file_id


def retrieve_openai_model(fine_tune_id):
    """
    retreive the status of model fine-tuning.
    :param fine_tune_id: the id for model
    """
    try:
        res = openai.FineTune.retrieve(id=fine_tune_id)
        id = res["id"]
        status = res["status"]  # the status will be succeeded when completed.
        fine_tuned_model = res["fine_tuned_model"]
        logger.info("id:{}, status:{}, fine_tuned_model:{}".format(
            id, status, fine_tuned_model))
    except Exception as ex:
        logger.error("openai.FineTune.retrieve() got an error, ex:{}".format(ex))
        status, fine_tuned_model = None, None
    return status, fine_tuned_model


def finetune_openai_model(
    training_file_id,
    suffix="detection",
    model="davinci",
    n_epochs=2,
    sleep_time_for_finetuning=60
):
    """
    fine-tune an openai model.
    :param training_file_id: the file id for training data
    :param suffix: suffix for the model name
    :param model: the baseline model name
    :param n_epochs: number of training epochs
    :param sleep_time_for_finetuning: sleep time in seconds
    """
    logger.info("## finetune_openai_model: {}".format(locals()))

    try:
        res = openai.FineTune.create(training_file=training_file_id,
                                    suffix=suffix,
                                    model=model,
                                    n_epochs=n_epochs)
        fine_tune_id = res["id"]
        status = res["status"]
        fine_tuned_model = res["fine_tuned_model"]
        logger.info("fine_tune_id:{}, status:{}, fine_tuned_model:{}".format(
            fine_tune_id, status, fine_tuned_model))
    except Exception as ex:
        return None

    fine_tuned_model = None
    if sleep_time_for_finetuning > 0:
        while status != "succeeded":
            time.sleep(sleep_time_for_finetuning)
            try:
                status, fine_tuned_model = retrieve_openai_model(fine_tune_id)
                logger.info("finetuning status:{}".format(status))
            except Exception as ex:
                logger.info(
                    "finetuning retrieve_openai_model ex:{}".format(ex))
        logger.info("finetuning status response:{}".format(res))
    return fine_tuned_model


def fine_tune_gpt3_model(
    df_train, 
    path_train_jsonl, 
    model_name,
    data_type="Message",
    positive_label="Spam",
    negative_label="Ham",
    column_text="text",
    column_label="label",
    fine_tune_context_sample_size=4
):
    """
    fine-tune a gpt3 model with the training dataset.
    :param df_train: df for training data
    :param path_train_jsonl: file path for training json file
    :param model_name: baseline model name
    :param data_type: data type
    :param positive_label: the value for positive label
    :param negative_label: the value for negative label
    :param column_text: the column for text
    :param column_label: the column for label
    :param fine_tune_context_sample_size: sample size for context prompt
    """
    training_file_id = upload_train_jsonl_file(
        path_train_jsonl,
        df_train,
        data_type=data_type,
        positive_label=positive_label,
        negative_label=negative_label,
        column_text=column_text,
        column_label=column_label,
        purpose="fine-tune",
        fine_tune_context_sample_size=fine_tune_context_sample_size
    )
    logger.info("training_file_id:{}".format(training_file_id))
    if training_file_id is None:
        return None

    fine_tuned_model_train = finetune_openai_model(
        training_file_id=training_file_id,
        suffix="detection_{}".format(data_type.lower()),
        model=model_name,
        n_epochs=2,
        sleep_time_for_finetuning=60
    )
    logger.info("fine_tuned_model_train:{}".format(fine_tuned_model_train))
    return fine_tuned_model_train


def classify_message(
    message,
    path_train_data="spam_data/experiment_1/train_2.csv",
    num_samples_in_prompt=2
):
    """
    classify the input message as spam or ham.
    spam_data/experiment_1/train_2.csv file has one ham and one spam samples.
    :param message: message data
    :param path_train_data: file path for few-shot samples
    :param num_samples_in_prompt: the number of samples in prompt
    """
    df_train = pd.read_csv(path_train_data, sep="\t")[:num_samples_in_prompt]

    context_prompt = generate_prompt_text(
        df_train, data_type="Message",
        positive_label="Spam", negative_label="Ham",
        column_text="text", column_label="label")

    query_text = QUERY_TEMPLATE.format(data_type="Message", text=message)
    prompt = context_prompt + query_text
    logger.info("prompt:{}".format(prompt))

    label = get_openai_completion(prompt, model_name="text-davinci-002")
    logger.info("label:{}".format(label))
    return label


def evaluate_gpt3_model(
    path_train_data,
    path_test_data,
    model_name="text-davinci-002",
    data_type="Message",
    column_text="text",
    column_label="label",
    positive_label="Spam",
    negative_label="Ham",
    fine_tune=False,
    fine_tuned_model="",
    fine_tune_context_sample_size=4,
    prompt_context_sample_size=3,
    sleep_time=1
):
    """
    evaluate a gpt3 model with training and test datasets.
    :param path_train_data: path for training dataset
    :param path_test_data: path for test dataset
    :param model_name: baseline gpt3 model name, text-davinci-002 for few-shot, davinci for fine-tuning
    :param data_type: data type
    :param column_text: the column for text
    :param column_label: the column for label
    :param positive_label: the value for positive label
    :param negative_label: the value for negative label
    :param fine_tune: True to fine-tune 
    :param fine_tuned_model: fine-tuned model name
    :param fine_tune_context_sample_size: the sample size for fine-tuning context prompt
    :param prompt_context_sample_size: the sample size for few-shot context prompt
    :param sleep_time: sleep time in seconds
    """
    df_train = pd.read_csv(path_train_data, sep="\t")
    logger.info("path_train_data:{}, df_train.shape:{}".format(
        path_train_data, df_train.shape))

    if fine_tune:
        path_train_jsonl = path_train_data + ".finetune.jsonl"
        model_name = fine_tune_gpt3_model(
            df_train, path_train_jsonl, model_name,
            data_type=data_type,
            positive_label=positive_label,
            negative_label=negative_label,
            column_text=column_text,
            column_label=column_label,
            fine_tune_context_sample_size=fine_tune_context_sample_size)
        if model_name is None:
            return None
        df_train = df_train.sample(prompt_context_sample_size, replace=False)
    elif fine_tuned_model:
        model_name = fine_tuned_model
        df_train = df_train.sample(prompt_context_sample_size, replace=False)

    context_prompt = generate_prompt_text(
        df_train, data_type=data_type,
        positive_label=positive_label, negative_label=negative_label,
        column_text=column_text, column_label=column_label)
    logger.info("context_prompt:{}".format(context_prompt))

    df_test = pd.read_csv(path_test_data, sep="\t")
    logger.info("path_test_data:{}, df_test.shape:{}".format(
        path_test_data, df_test.shape))

    QUERY_TEMPLATE = "\n{data_type}: {text}\nLabel:"
    y_test, y_pred = [], []
    count_correct = 0
    for ix, row in df_test.iterrows():
        text = row[column_text]
        label = row[column_label]
        logger.info("{}.text:{}".format(ix, text))
        query_text = QUERY_TEMPLATE.format(data_type=data_type, text=text)
        prompt = context_prompt + query_text
        completion = get_openai_completion(prompt, model_name=model_name)
        if completion is not None:
            y_test.append(label == positive_label)
            y_pred.append(completion == positive_label)
            count_correct += 1 if label == completion else 0
            if sleep_time > 0:
                time.sleep(sleep_time)
            logger.info("label:{}, completion:{}, count_correct:{}".format(
                label, completion, count_correct))

    return classification_report(y_test, y_pred, output_dict=True)
