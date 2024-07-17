import collections
import logging
import math
from pathlib import Path
import re

from .codebleu.bleu import corpus_bleu as ngram_corpus_bleu
from .codebleu.weighted_ngram_match import corpus_bleu as weighted_ngram_corpus_bleu
from .codebleu.syntax_match import corpus_syntax_match
from .codebleu.dataflow_match import  corpus_dataflow_match

class Bleu:
    @staticmethod
    def get_ngrams(segment, max_order):
        """
            Extracts all n-grams upto a given maximum order from an input segment.
            Args:
                segment: text segment from which n-grams will be extracted.
                max_order: maximum length in tokens of the n-grams returned by this methods.
            Returns:
                The Counter containing all n-grams upto max_order in segment
                with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    @staticmethod
    def compute_bleu(references, candidates, max_order=4, smooth=False):
        """
        Computes BLEU score of translated segments against one or more references.
        Args:
            references: list of references for each candidate. Each reference should be tokenized into a list of tokens.
            candidates: list of candidates to score. Each candidate should be tokenized into a list of tokens.
            max_order: Maximum n-gram order to use when computing BLEU score.
            smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
            3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        length = len(references)
        reference_length = 0
        candidate_length = 0
        for i in range(length):
            reference_length += len(references[i])
            candidate_length += len(candidates[i])
            reference_ngram_counts = Bleu.get_ngrams(references[i],max_order)
            candidate_ngram_counts = Bleu.get_ngrams(candidates[i],max_order)
            overlap = reference_ngram_counts & candidate_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1,max_order+1):
                possible_matches = len(candidates[i]) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0,max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(candidate_length) / reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp
        return bleu



class CodeBleu:
    @staticmethod
    def make_weights(reference_tokens, keyword_list):
        return {token:1 if token in keyword_list else 0.2 for token in reference_tokens}
    
    @staticmethod
    def compute_codebleu(references, candidates, lang="python", alpha=0.25, beta=0.25, gamma=0.25, theta=0.25):

        # preprocess inputs
        references = [[ref.strip()] for ref in references] 
        candidates = [cand.strip() for cand in candidates] 

        # calculate ngram match (BLEU)
        tokenized_cands = [x.split() for x in candidates]
        tokenized_refs = [[x.split() for x in reference] for reference in references]
        ngram_match_score = ngram_corpus_bleu(tokenized_refs, tokenized_cands)

        # calculate weighted ngram match
        keywords = [x.strip() for x in (Path(__file__).parent / f"codebleu/keywords/{lang}.txt").open('r', encoding='utf-8').readlines()]
        tokenized_refs_with_weights = [[[reference_tokens, CodeBleu.make_weights(reference_tokens, keywords)]\
                    for reference_tokens in reference] for reference in tokenized_refs]
        weighted_ngram_match_score = weighted_ngram_corpus_bleu(tokenized_refs_with_weights,tokenized_cands)

        # calculate syntax match
        syntax_match_score = corpus_syntax_match(references, candidates, lang)

        # calculate dataflow match
        dataflow_match_score = corpus_dataflow_match(references, candidates, lang)

        logging.info('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                        format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

        code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

        return code_bleu_score
        # print('CodeBLEU score: ', code_bleu_score)

class LSPHit:
    @staticmethod
    def hit(references, candidates, lsp_point="<lsp>", record_f=None):
        total_count = 0
        hit_count = 0
        for ref, cand in zip(references, candidates):
            for mobj in re.finditer(r"(%s)(\w+)(\W|$)" % re.escape(lsp_point), ref):
                key_ele = mobj.group(2).strip()
                if len(key_ele) == 0:
                    continue
                total_count += 1
                if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), cand):
                    hit_count += 1
                    if record_f is not None:
                        record_f.write(f"(√) {key_ele}\n")
                elif record_f is not None:
                    record_f.write(f"(x) {key_ele}\n")
        logging.info(f"total lsp count: {total_count}, hit lsp count: {hit_count}")
        return hit_count/total_count if total_count > 0 else 0
                