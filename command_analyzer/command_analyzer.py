import json
import pandas as pd
import argparse
import logging

from prompt_data import get_prompt_for_desc_from_cmd_tag
from prompt_data import get_prompt_for_cmd_from_tag_desc
from prompt_data import get_prompt_for_combined_desc
from prompt_data import preprocess_tags_str
from prompt_data import generate_text_list_with_prompt
from similarity import get_sorted_similarity_score_list
from similarity import get_ngrams_bleu_similarity_score
from similarity import get_semantic_similarity_score


logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)    
logger = logging.getLogger(__name__)


MAX_CMD_LEN = 200

ENGINE_CMD2DESC = "code-davinci-002"
ENGINE_DESC2CMD = "code-davinci-002"
ENGINE_COMBINE_DESC = "text-davinci-002"
ENGINE_EMBEDDINGS = "text-similarity-babbage-001"

ENGINE_TEMPERATURE = 0.7


def generate_desc_list_from_cmd_tag(
    cmd, 
    tags, 
    include_tag=True, 
    include_prefix=False, 
    engine=ENGINE_CMD2DESC,
    temperature=0.7, 
    n=5, 
    max_cmd_len=MAX_CMD_LEN
):
    """
    generates a list of descriptions from the command and tag info.
    :param cmd: command line data
    :param tags: "," seperated tags data
    :param include_tag: True to include tags in prompt
    :param include_prefix: True to include support examples in prompt
    :param engine: openai engine
    :param temperature: temperature to control randomness of engine
    :param n: number of engine outputs
    :param max_cmd_len: max data length for command line data
    """
    prompt = get_prompt_for_desc_from_cmd_tag(
        cmd, tags, max_cmd_len=max_cmd_len, 
        include_tag=include_tag, include_prefix=include_prefix)
    desc_list = generate_text_list_with_prompt(
        prompt, engine=engine, temperature=temperature, n=n)
    return desc_list


def generate_cmd_list_from_tag_desc(
    cmd, 
    tags, 
    desc, 
    include_tag=True, 
    include_prefix=False, 
    engine=ENGINE_DESC2CMD,
    temperature=0.7, 
    n=1, 
    max_cmd_len=MAX_CMD_LEN
):
    """
    generates a list of command lines from the description and tag info.
    :param cmd: command line data
    :param tags: "," seperated tags data
    :param desc: description for the command line
    :param include_tag: True to include tags in prompt
    :param include_prefix: True to include support examples in prompt
    :param engine: openai engine
    :param temperature: temperature to control randomness of engine
    :param n: number of engine outputs
    :param max_cmd_len: max data length for command line data
    """
    first_token_as_cmd = cmd.split()[0]
    prompt = get_prompt_for_cmd_from_tag_desc(
        tags, desc, first_token_as_cmd, max_cmd_len=max_cmd_len, 
        include_tag=include_tag, include_prefix=include_prefix)
    cmd_list = generate_text_list_with_prompt(
        prompt, engine=engine, temperature=temperature, n=n)
    return cmd_list


def generate_combined_desc_from_cmd_desc(
    cmd, 
    desc1, 
    desc2, 
    engine=ENGINE_COMBINE_DESC,
    temperature=0.7, 
    max_cmd_len=MAX_CMD_LEN
):
    """
    generates a combined descriptin from two descriptions.
    :param cmd: command line data
    :param desc1: the first description
    :param desc2: the second description
    :param engine: openai engine
    :param temperature: temperature to control randomness of engine
    :param max_cmd_len: max data length for command line data
    """
    prompt = get_prompt_for_combined_desc(
        cmd, desc1, desc2, max_cmd_len=max_cmd_len)
    desc = generate_text_list_with_prompt(
        prompt, engine=engine, temperature=temperature, n=1)[0]
    return desc


def generate_sorted_desc_list_from_cmd_tag(
    cmd, 
    tags, 
    include_tag=True, 
    include_prefix=False,
    weight_desc_score=.0, 
    weight_tags_score=.0, 
    desc_size=5, 
    cmd_size=1,
    engine_cmd2desc=ENGINE_CMD2DESC,
    engine_desc2cmd=ENGINE_DESC2CMD, 
    engine_embeddings=ENGINE_EMBEDDINGS,
    temperature=ENGINE_TEMPERATURE,
    max_cmd_len=MAX_CMD_LEN
):
    """
    generates a list of descriptions sorted by similarity scores.
    step1. generate a list of descs from cmd and tags
    step2. generate a list of cmds from tags and desc
    step3. sort descs by similarity scores

    :param cmd: command line data
    :param tags: "," seperated tags data
    :param include_tag: True to include tags in prompt
    :param include_prefix: True to include support examples in prompt
    :param weight_desc_score: the weight for description score
    :param weight_tags_score: the weight for tags score
    :param desc_size: the number of output descriptions 
    :param cmd_size: the number of output command lines
    :param engine_cmd2desc: the engine for command to description
    :param engine_desc2cmd: the engine for description to command
    :param engine_embeddings: the engine for text embeddings
    :param temperature: temperature to control randomness of engine
    :param max_cmd_len: max data length for command line data
    """
    tags = preprocess_tags_str(tags)
    logger.info(f"tags={tags}")

    desc_list = generate_desc_list_from_cmd_tag(
        cmd, tags, include_tag=include_tag, include_prefix=include_prefix, 
        engine=engine_cmd2desc, temperature=temperature, 
        n=desc_size, max_cmd_len=max_cmd_len)
    if len(desc_list) == 0:
        return []
    baseline_description = desc_list[0]

    cmd_first_token = cmd.split()[0]
    desc_cmd_list = []
    for desc in desc_list: 
        generated_cmd = generate_cmd_list_from_tag_desc(
            cmd, tags, desc, include_tag=include_tag, 
            include_prefix=include_prefix, n=cmd_size, 
            engine=engine_desc2cmd, temperature=temperature,
            max_cmd_len=max_cmd_len)[0]
        #append the cmd_first_token + the first generated text
        generated_cmd = cmd_first_token + generated_cmd
        desc_cmd_list.append((desc, generated_cmd))
    desc_cmd_list = get_sorted_similarity_score_list(
        cmd, tags, 
        desc_cmd_list, engine=engine_embeddings,
        weight_desc_score=weight_desc_score, 
        weight_tags_score=weight_tags_score, max_cmd_len=max_cmd_len)
    return desc_cmd_list, baseline_description


def generate_descriptions_from_cmd_tags(
    cmd, 
    tags=None, 
    n=2,
    combine_descriptions=True,
    engine_cmd2desc=ENGINE_CMD2DESC,
    engine_desc2cmd=ENGINE_DESC2CMD,
    engine_embeddings=ENGINE_EMBEDDINGS,
    engine_combine_desc=ENGINE_COMBINE_DESC,
    temperature=ENGINE_TEMPERATURE,  
    max_cmd_len=MAX_CMD_LEN
):
    """
    generates a combined descripotion from a list of descriptions.
    :param cmd: command line data
    :param tags: "," seperated tags data
    :param n: the number of output descriptions 
    :param combine_descriptions: True to combine two descriptions
    :param engine_cmd2desc: the engine for command to description
    :param engine_desc2cmd: the engine for description to command
    :param engine_embeddings: the engine for text embeddings
    :param temperature: temperature to control randomness of engine
    :param max_cmd_len: max data length for command line data
    """
    logger.info("generate_descriptions_from_cmd_tags:{}".format(locals()))

    if tags:
        include_tag = 1
        weight_desc_score=0.3
        weight_tags_score=0.2
    else:
        include_tag = 0
        weight_desc_score=0.5
        weight_tags_score=0.0
        
    desc_cmd_items, baseline_description = generate_sorted_desc_list_from_cmd_tag(
        cmd, tags, include_tag=include_tag, include_prefix=1,
        weight_desc_score=weight_desc_score, weight_tags_score=weight_tags_score,
        desc_size=5, cmd_size=1, 
        engine_cmd2desc=engine_cmd2desc, engine_desc2cmd=engine_desc2cmd,
        engine_embeddings=engine_embeddings, temperature=temperature,
        max_cmd_len=max_cmd_len)

    baseline_description = "The command" + baseline_description

    best_candidates = []
    first_desc, second_desc = '', ''
    for ix,item in enumerate(desc_cmd_items[:n]):
        score, _score_code, _score_desc, _score_tags, cmd, generated_cmd, desc = item[:]
        candidate_data = {"score":score, "desc":desc, "generated_cmd":generated_cmd}
        if ix == 0:
            first_desc = desc
        else:
            second_desc = desc
        logger.info(candidate_data)
        best_candidates.append(candidate_data)

    if first_desc and second_desc:
        if combine_descriptions:
            combined_desc = generate_combined_desc_from_cmd_desc(
                cmd, first_desc, second_desc, 
                engine=engine_combine_desc, temperature=temperature,
                max_cmd_len=max_cmd_len)
            description = "The command" + combined_desc
        else:
            description = "The command" + first_desc
    else:
        description = "The command" + first_desc
    logger.info("\n"+description+"\n")
    
    return description, baseline_description, best_candidates


def generate_description(
    cmd, 
    tags=None,
    combine_descriptions=True,
    engine_cmd2desc=ENGINE_CMD2DESC,
    engine_desc2cmd=ENGINE_DESC2CMD,
    temperature=ENGINE_TEMPERATURE
):
    """
    generates a description from a command line and tag info.
    :param cmd: command line data
    :param tags: "," seperated tags data
    :param combine_descriptions: True to combine two descriptions
    :param engine_cmd2desc: the engine for command to description
    :param engine_desc2cmd: the engine for description to command
    :param temperature: temperature to control randomness of engine
    """
    description, baseline_description, best_candidates = generate_descriptions_from_cmd_tags(
        cmd, tags=tags, n=2,
        combine_descriptions=combine_descriptions,
        engine_cmd2desc=engine_cmd2desc, engine_desc2cmd=engine_desc2cmd,
        temperature=temperature
    )

    logger.info("cmd:\n{}".format(cmd))
    logger.info("tags:\n{}".format(tags))
    logger.info("description:\n{}".format(description))
    logger.info("baseline_description:\n{}".format(baseline_description))
    logger.info("back-translated_cmd:\n{}".format(best_candidates[0]["generated_cmd"]))
    return description, baseline_description, best_candidates


def evaluate_approaches(
    path_output_json, 
    path_input_json,
    engine_cmd2desc=ENGINE_CMD2DESC,
    engine_desc2cmd=ENGINE_DESC2CMD,
    temperature=ENGINE_TEMPERATURE,
    combine_descriptions=True,
    offset=0, 
    limit=0
):
    """
    evaluates baseline and back-translation approaches using a test dataset.
    :param path_output_json: file path for input data
    :param path_input_json: file path for output data
    :param cmd: command line data
    :param tags: "," seperated tags data
    :param engine_cmd2desc: the engine for command to description
    :param engine_desc2cmd: the engine for description to command
    :param temperature: temperature to control randomness of engine
    :param combine_descriptions: True to combine two descriptions
    :param offset: the offset of input data for partial testing
    :param limit: the limit of input data for partial testing
    """
    logger.info("evaluate_approaches:{}".format(locals()))

    with open(path_input_json) as fr:
        items = json.load(fr)

    if offset>0:
        items = items[offset:]
    if limit>0:
        items = items[:limit]
    logger.info("number of items in input file:{}".format(len(items)))

    results = []
    for ix, item in enumerate(items):
        logger.info("======== {}".format(ix))

        cmd = item["cmd"]
        tags = item["tags"]
        gold_description = item["gold_reference_description"]
        
        generated_description, baseline_description, _best_candidates = generate_description(
            cmd, 
            tags=tags,
            combine_descriptions=combine_descriptions,
            engine_cmd2desc=engine_cmd2desc,
            engine_desc2cmd=engine_desc2cmd,
            temperature=temperature
        )

        item["generated_description"] = generated_description
        item["baseline_description"] = baseline_description
        
        candidate_list = [generated_description, baseline_description]

        ngram_bleu_scores = get_ngrams_bleu_similarity_score(gold_description, candidate_list)
        #ngram_bleu_scores_list.append(ngram_bleu_scores) 
        item["ngram_bleu_scores"] = {
            "generated_description_score":ngram_bleu_scores[0], "baseline_description_score":ngram_bleu_scores[1]
        }

        semantic_similarity_scores = get_semantic_similarity_score(gold_description, candidate_list, engine=ENGINE_EMBEDDINGS)
        #semantic_similarity_scores_list.append(semantic_similarity_scores)
        item["semantic_similarity_scores"] = {
            "generated_description_score":semantic_similarity_scores[0], "baseline_description_score":semantic_similarity_scores[1]
        }

        results.append(item)

        logger.info("cmd:{}".format(cmd))
        logger.info("tags:{}".format(tags))
        logger.info("gold_description:{}".format(gold_description))
        logger.info("generated_description:{}".format(generated_description))
        logger.info("baseline_description:{}".format(baseline_description))

        logger.info("ngram_bleu_scores:{}".format(ngram_bleu_scores))
        logger.info("semantic_similarity_scores:{}".format(semantic_similarity_scores))

    #save the outputs
    with open(path_output_json, "wt") as fw:
        json.dump(results, fw, indent=2)

    #store the mean and std values for similarity scores
    df_bleu = pd.DataFrame([item["ngram_bleu_scores"] for item in results]).add_prefix("ngram_bleu_")
    df_semantic = pd.DataFrame([item["semantic_similarity_scores"] for item in results]).add_prefix("semantic_similarity_")
    df = df_bleu.join(df_semantic)
    df_mean_std = df.agg(["mean", "std"])
    logger.info("df_mean_std:{}".format(df_mean_std))
    path_output_score_csv = path_output_json + "_scores.csv"
    df_mean_std.to_csv(path_output_score_csv)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to description")

    parser.add_argument(
        "--run_type",
        help="denerate_desc or evaluate_approaches",
        default="generate_desc"
    )
    parser.add_argument(
        "--cmd",
        help="command line data",
    )
    parser.add_argument(
        "--tags",
        help="',' seperated tags for example, win_mimikatz_command_line,win_suspicious_execution_path ",
        default=""
    )
    parser.add_argument(
        "--combine_descriptions",
        action="store_true",
        dest="combine_descriptions",
        help="to combine two descriptions as the final description",
        default=True
    )
    parser.add_argument(
        "--no_combine_descriptions",
        action="store_false",
        dest="combine_descriptions",
        help="not to combine two descriptions as the final description",
        default=False
    )

    parser.add_argument(
        "--engine_cmd2desc",
        help="gpt3 model for command to description",
        default=ENGINE_CMD2DESC
    )
    parser.add_argument(
        "--engine_desc2cmd",
        help="gpt3 model for description to command",
        default=ENGINE_DESC2CMD
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature ",
        default=ENGINE_TEMPERATURE
    )

    parser.add_argument(
        "--path_output_json",
        help="path for output json file",
    )
    parser.add_argument(
        "--path_input_json",
        help="path for input json file",
    )
    parser.add_argument(
        "--offset",
        type=int,
        help="path for input json file",
        default=0
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="path for input json file",
        default=0
    )

    args = parser.parse_args()
    if args.run_type == "generate_desc":
        generate_description(
            args.cmd, 
            args.tags,
            args.combine_descriptions,
            engine_cmd2desc=args.engine_cmd2desc,
            engine_desc2cmd=args.engine_desc2cmd
        )
    else:
        evaluate_approaches(
            args.path_output_json, 
            args.path_input_json, 
            engine_cmd2desc=args.engine_cmd2desc,
            engine_desc2cmd=args.engine_desc2cmd,
            temperature=args.temperature,
            combine_descriptions=args.combine_descriptions,
            offset=args.offset,
            limit=args.limit
        )        
