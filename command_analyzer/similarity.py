import numpy as np
import logging
from nltk.translate.bleu_score import sentence_bleu
from openai.embeddings_utils import get_embeddings


logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)    
logger = logging.getLogger(__name__)


def get_embedding_similarity_score_list(
    cmd, 
    tags, 
    items, 
    max_cmd_len=200,
    weight_desc_score=0.3, 
    weight_tags_score=0.2, 
    engine="text-similarity-babbage-001"
):
    """
    returns a weighted score from cosine similarity scores.
    :param cmd: command line data
    :param tags: tags data
    :param items: contains a list of (desc, generated_cmd)
    :param max_cmd_len: max length of command line data
    :param weight_desc_score: the weight for description score
    :param weight_tags_score: the weight for tags score
    :param engine: engine for similarity score 
    """
    code_items = [cmd[:max_cmd_len]] + [item[1] for item in items]
    if weight_tags_score>0:
        code_items += [tags]
    code_matrix = get_embeddings(code_items, engine=engine)
    if weight_tags_score>0:
        tags_emb = code_matrix[-1]

    if weight_desc_score>0:
        desc_items = [item[0] for item in items]
        desc_matrix = get_embeddings(desc_items, engine=engine)

    reference_code_emb = code_matrix[0]
    score_list = []
    for ix in range(len(items)):
        desc, generated_cmd = items[ix]
        #get a cosine similarity score between two embeddings vectors
        code_score = np.dot(reference_code_emb, code_matrix[1+ix])
        #get a weighted score from 3 scores
        if weight_desc_score>0 and weight_tags_score>0:
            desc_score = np.dot(reference_code_emb, desc_matrix[ix])
            tags_score = np.dot(tags_emb, desc_matrix[ix])
            score = (1-weight_desc_score-weight_tags_score) * code_score + weight_desc_score * desc_score + weight_tags_score * tags_score
        elif weight_desc_score>0:
            desc_score = np.dot(reference_code_emb, desc_matrix[ix])
            tags_score = .0
            score = (1-weight_desc_score) * code_score + weight_desc_score * desc_score
        elif weight_tags_score>0:
            desc_score = .0
            tags_score = np.dot(reference_code_emb, desc_matrix[ix])
            score = (1-weight_tags_score) * code_score + weight_tags_score * tags_score
        else:
            desc_score = .0
            tags_score = .0
            score = code_score
        logger.info("===\n{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            score, code_score, desc_score, tags_score))
        logger.info(desc)
        logger.info(generated_cmd)
        score_list.append((score, code_score, desc_score, tags_score, cmd, generated_cmd, desc))
    return score_list


def get_sorted_similarity_score_list(
    cmd, 
    tags, 
    items, 
    engine="text-similarity-babbage-001", 
    weight_desc_score=0.3, 
    weight_tags_score=0.2, 
    max_cmd_len=200
):
    """
    return a list of description list sorted similarity scores.
    :param cmd: command line data
    :param tags: tags data
    :param items: contains a list of (desc, generated_cmd)
    :param engine: engine for similarity score 
    :param weight_desc_score: the weight for description score
    :param weight_tags_score: the weight for tags score
    :param max_cmd_len: max length of command line data
    """
    score_list = get_embedding_similarity_score_list(
        cmd, tags, items,
        weight_desc_score=weight_desc_score, 
        weight_tags_score=weight_tags_score,
        engine=engine, max_cmd_len=max_cmd_len)

    return sorted(score_list, key=lambda x:x[0], reverse=True)


def get_ngrams_bleu_similarity_score(
    reference, 
    candidate_list
):
    """
    returns similarity scores using sentence_bleu.
    :param reference: reference text
    :param candidate_list: a list of candidates 
    """
    score_list = []
    for candidate in candidate_list:
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        score = sentence_bleu(reference_tokens, candidate_tokens,
            weights=(0.5, 0.5, 0., 0.))
        score_list.append(score)
    return score_list


def get_semantic_similarity_score(
    reference, 
    candidate_list, 
    engine="text-similarity-babbage-001"
):
    """
    return similarity scores using cosine similarity with gpt3 embeddings.
    :param reference: reference text
    :param candidate_list: a list of candidates 
    :param engine: engine for similarity score 
    """
    items = [reference] + candidate_list
    embeddings_list = get_embeddings(items, engine=engine)
    reference_emb = embeddings_list[0]
    score_list = []
    for candidate_embeddings in embeddings_list[1:]:
        #get a cosine similarity score
        score = np.dot(reference_emb, candidate_embeddings)
        score_list.append(score)
    return score_list
