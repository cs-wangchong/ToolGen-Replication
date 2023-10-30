from pathlib import Path
import logging
import ast
import re

from tqdm import tqdm

import jedi
import torch
from transformers import (
    LogitsProcessorList,
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from coder.generator import Generator
from coder.data_augmentor import DataAugmentor
from coder.utils.trie import Trie
from coder.utils.metric_utils import Bleu, CodeBleu
from coder.constants import *

def clean_lsp(code:str):
    code = re.sub(r" ?%s ?" % re.escape(PLM_LSP_POINT), "", code)
    return code

def clean_str(code):
    code = re.sub(r"'(.*?)'", "", code)
    code = re.sub(r'"(.*?)"', "", code)
    return code.strip()

class LSPCoder:
    def __init__(self, generator:Generator):
        self.generator = generator
        self.device = generator.device
        self.jedi_pj = None
        
    def evaluate_with_lsp(
            self,
            eval_set,
            batch_size=16,
            max_len=256,
            beam_size=2,
            cand_num=3,
            lsp_threshold=0.8,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6,
            repetition_penalty=1.0,
            record_file=None
        ):
        self.generator.model.eval()
        queries, predictions, expectations = [], [], []
        batch_ranges = list(zip(range(0, len(eval_set), batch_size), range(batch_size, len(eval_set)+batch_size, batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Evaluation"):
                batch = eval_set[beg:end]
                _queries, _predictions, _expectations = [], [], []
                for repo, file_path, desc, code in batch:
                    logging.info(f"====================================")
                    logging.info(f"[EXPECTATION]\n{code.strip()}")
                    _queries.append(desc)
                    _expectations.append(code)
                    self.update_lsp_project(repo)
                    context, line, column = self.get_lsp_context(file_path, code)
                    if context is None:
                        output = self.generator.generate([desc], beam_size=beam_size, max_len=max_len, repetition_penalty=repetition_penalty)[0]
                    else:
                        output = self.generate_with_lsp(desc, file_path, context, line, column, max_len, beam_size, cand_num, lsp_threshold, token_threshold, token_k, temperature, repetition_penalty)
                    _predictions.append(output)
                queries.extend(_queries)
                predictions.extend(_predictions)
                expectations.extend(_expectations)
        
        total_count = 0
        hit_count = 0
        infos = []
        for query, pred, expt in zip(queries, predictions, expectations):
            info = f"======================\n[PROMPT]:\n{query}\n[EXPECTATION]:\n{expt}\n[PREDICTION]:\n{pred}\n"
            signature = query.split("Function Signature:")[-1].strip()
            key_eles = set()
            for mobj in re.finditer(r"%s(\w+)(\W|$)" % re.escape(PLM_LSP_POINT), expt):
                key_ele = mobj.group(1).strip()
                if len(key_ele) == 0:
                    continue
                if re.search(r"\s+%s\s*=" % re.escape(key_ele), expt):
                    continue
                if re.search(r"as\s+%s\s*:" % re.escape(key_ele), expt):
                    continue
                if re.search(r"for\s+\W*%s\W*\s+in" % re.escape(key_ele), expt):
                    continue
                if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), signature):
                    continue
                key_eles.add(key_ele)
            for key_ele in key_eles:
                total_count += 1
                if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), clean_str(pred)):
                    hit_count += 1
                    info = f"{info}(√) {key_ele}\n"
                else:
                    info = f"{info}(x) {key_ele}\n"
            infos.append(info)
        logging.info(f"total lsp count: {total_count}, hit lsp count: {hit_count}")
        lsp_hit = hit_count/total_count if total_count > 0 else 0

        predictions = [clean_lsp(code) for code in predictions]
        expectations = [clean_lsp(code) for code in expectations]
        codebleu = CodeBleu.compute_codebleu(expectations, predictions)

        predictions = [self.generator.tokenizer.tokenize(code) for code in predictions]
        expectations = [self.generator.tokenizer.tokenize(code) for code in expectations]
        bleu = Bleu.compute_bleu(expectations, predictions, smooth=True)

        if record_file is not None:
            with Path(record_file).open("w") as f:
                f.write("\n".join(infos))

        return bleu, codebleu, lsp_hit
    
    def _init_decoding(self, source_ids, attention_mask, beam_size=2):
        decoding_ids = torch.tensor([self.generator.model.generation_config.decoder_start_token_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs: Seq2SeqLMOutput = self.generator.model(
                input_ids = source_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_ids.unsqueeze(0),
            )
        encoder_outputs = BaseModelOutput(
            last_hidden_state=outputs.encoder_last_hidden_state,
            hidden_states=None,
            attentions=None,
        )
        logits = outputs.logits[:,-1,:]
        probs = torch.softmax(logits, -1).view(-1)
        beam_probs, best_idx = probs.topk(beam_size, 0, True, True)
        beam_decoding_ids = []
        for i in range(beam_size):
            beam_decoding_ids.append(torch.cat([decoding_ids, best_idx[i:i+1]], -1))
        beam_past_key_values = [outputs.past_key_values for _ in range(beam_size)]
        beam_scores = torch.log(beam_probs)
        return encoder_outputs, beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores
    
    def _beam_advance(
            self,
            encoder_outputs,
            attention_mask,
            decoding_inputs,
            past_key_values,
            cur_score,
            logits_processor,
            beam_size=2
        ):
        with torch.no_grad():
            outputs: Seq2SeqLMOutput = self.generator.model(
                encoder_outputs = encoder_outputs,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_inputs.unsqueeze(0)[:,-1:],
                past_key_values=past_key_values
            )
        logits = outputs.logits[:,-1,:]
        probs = torch.softmax(logits, -1)
        log_probs = torch.log(probs)
        log_probs = logits_processor(decoding_inputs.unsqueeze(0), log_probs)
        probs = probs.view(-1)
        log_probs = log_probs.view(-1)
        flat_scores = torch.ones_like(log_probs) * cur_score + log_probs
        best_scores, best_idx = flat_scores.topk(beam_size, 0, True, True)
        best_probs = probs[best_idx]

        best_decoding_ids = [torch.cat([decoding_inputs, best_idx[i:i+1]], -1) for i in range(beam_size)]
        best_past_key_values = [outputs.past_key_values for i in range(beam_size)]
        return best_decoding_ids, best_past_key_values, best_probs, best_scores
    
    def _lsp_expand(
            self,
            all_cands,
            encoder_outputs,
            attention_mask,
            decoding_inputs,
            past_key_values,
            logits_processor,
            expand_batch_size=16,
            cand_num=3,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6
        ):
        
        lsp_decoder_input_ids = decoding_inputs.unsqueeze(0).repeat(1, 1)
        lsp_past_key_values =  past_key_values
        lsp_probs = [0]
        lsp_scores =  [0]

        trie = Trie()
        for cand in all_cands:
            trie.insert([self.generator.tokenizer._convert_token_to_id(t) for t in self.generator.tokenizer.tokenize(cand)])

        # calculate scores for candidates
        pending_nodes = [trie.root]
        all_cands_with_scores = []

        step = 0
        while True:
            new_pending_nodes = []
            children_list = []
            for i, node in enumerate(pending_nodes):
                if node.is_end_of_sequence and node.is_valid:
                    _decoding_ids = lsp_decoder_input_ids[i]
                    _past_key_values = [[x[i:i+1,:,:,:] for x in y] for y in lsp_past_key_values]
                    _score = lsp_scores[i]
                    _prob = lsp_probs[i]
                    cand = self.generator.tokenizer.decode(_decoding_ids[-step:], skip_special_tokens=True)
                    all_cands_with_scores.append((cand, step, _decoding_ids, _past_key_values, _prob, _score))
                elif len(node.get_children()) > 0:
                    new_pending_nodes.append(node)
                    children_list.append(node.get_children())
            pending_nodes = new_pending_nodes
            step += 1
            if len(pending_nodes) == 0:
                break
            
            logits_list = []
            past_kvs_list = []
            step_size = lsp_decoder_input_ids.size(0)
            for beg in range(0, step_size, expand_batch_size):
                end = min(beg + expand_batch_size, step_size)
                with torch.no_grad():
                    tmp_encoder_outputs = BaseModelOutput(
                        last_hidden_state=encoder_outputs.last_hidden_state.repeat(end-beg, 1, 1),
                        hidden_states=None,
                        attentions=None,
                    )
                    tmp_attention_mask = attention_mask.repeat(end-beg, 1, 1)
                    outputs = self.generator.model(
                        encoder_outputs = tmp_encoder_outputs,
                        attention_mask = tmp_attention_mask,
                        decoder_input_ids = lsp_decoder_input_ids[beg:end,-1:],
                        past_key_values = [[x[beg:end] for x in y] for y in lsp_past_key_values]
                    )
                logits_list.append(outputs.logits[:,-1,:])
                past_kvs_list.append(outputs.past_key_values)
            logits = torch.cat(logits_list, 0)
            lsp_past_key_values = [[torch.cat(xs, 0) for xs in zip(*ys)] for ys in zip(*past_kvs_list)]
            tau = torch.ones_like(logits, device=self.device)
            tau_x, tau_y = [], []
            for idx, children in enumerate(children_list):
                tau_x.extend([idx] * len(children))
                tau_y.extend([node.key for node in children])
            tau[tau_x, tau_y] = 1 / temperature
            logits *= tau
            probs = torch.softmax(logits, -1)
            logprobs = torch.log(probs)
            # probs = logits_processor(tmp_decoder_input_ids, probs)

            next_token_ids_list = []
            next_token_probs_list = []
            next_token_logprobs_list = []
            new_pending_nodes = []
            for k, (_children, _probs, _logprobs) in enumerate(zip(children_list, probs, logprobs)):
                _children.sort(key=lambda child_node: _probs[child_node.key].item(), reverse=True)
                logging.info('\t' * step + f"group: {i}")
                for child_node in _children:
                    logging.info('\t' * step + f"token: {self.generator.tokenizer.convert_ids_to_tokens([child_node.key])[0]}, prob: {_probs[child_node.key].item()}")
                _children = _children[:cand_num]
                if token_k:
                    _topk_probs, _topk_idxs = probs.topk(token_k, -1, True, True)
                    _topk_idxs = {_idx.item() for _idx in _topk_idxs}
                    topk_tokens_with_probs = [(self.generator.tokenizer.convert_ids_to_tokens([_idx.item()])[0], round(_prob.item(), 4)) for _prob, _idx in zip(_topk_probs, _topk_idxs)]
                    topk_tokens_with_probs.sort(key=lambda x: x[-1], reverse=True)
                
                next_token_ids, next_token_probs, next_token_logprobs = [], [], []
                for child_node in _children:
                    if _probs[child_node.key].item() < token_threshold or (token_k and child_node.key not in _topk_idxs):
                        pending_nodes[k].remove_child(child_node.key)
                        continue
                    # child_node.set_score(_probs[child_node.key].item())
                    next_token_ids.append(child_node.key)
                    next_token_probs.append(_probs[child_node.key].item())
                    next_token_logprobs.append(_logprobs[child_node.key].item())
                    new_pending_nodes.append(child_node)
                next_token_ids_list.append(next_token_ids)
                next_token_probs_list.append(next_token_probs)
                next_token_logprobs_list.append(next_token_logprobs)
            
            pending_nodes = new_pending_nodes
            total_num = sum(len(next_token_ids) for next_token_ids in next_token_ids_list)
            if total_num == 0:
                break

            new_lsp_decoder_input_ids = []
            for i, next_token_ids in enumerate(next_token_ids_list):
                if len(next_token_ids) == 0:
                    continue
                expanded_rows = lsp_decoder_input_ids[i:i+1].repeat(len(next_token_ids), 1)
                next_ids = torch.tensor([[t] for t in next_token_ids], dtype=torch.long, device=self.device)
                new_lsp_decoder_input_ids.append(torch.cat((expanded_rows, next_ids), -1))
            lsp_decoder_input_ids = torch.cat(new_lsp_decoder_input_ids, 0)   
            new_lsp_past_key_values = []
            for y in lsp_past_key_values:
                new_y = []
                for x in y:
                    new_x = []
                    for i, next_token_ids in enumerate(next_token_ids_list):
                        if len(next_token_ids) == 0:
                            continue
                        expanded_rows = x[i:i+1].repeat(len(next_token_ids), 1, 1, 1)
                        new_x.append(expanded_rows)
                    new_y.append(torch.cat(new_x, 0))
                new_lsp_past_key_values.append(new_y)
            lsp_past_key_values = new_lsp_past_key_values
            lsp_probs = [prob for next_token_probs in next_token_probs_list for prob in next_token_probs]
            lsp_scores = [score + logprob for score, next_token_logprobs in zip(lsp_scores, next_token_logprobs_list) for logprob in next_token_logprobs]
        
        if len(all_cands_with_scores) == 0:
            logging.info("\t\tNo valid candidates")
            return [], [], None, None
        all_cands_with_scores.sort(key=lambda x: x[-1], reverse=True)
        logging.info(f"\t\tCandidates with scores:")
        for r, (_cand, _len, _, _, _, _score) in enumerate(all_cands_with_scores, 1):
            logging.info(f"\t\t\tNo. {r}: {repr(_cand)}, score: {_score}, length: {_len}")
            # for _step, (cand_token, cand_prob, topk_tokens) in enumerate(cand2steps[_cand]):
            #     print(f"\t\t\tstep: {_step} | {cand_token} | {cand_prob} | {topk_tokens}")
        _, _, lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores = zip(*all_cands_with_scores[:cand_num])
        lsp_probs = torch.tensor(lsp_probs, dtype=torch.float, device=self.device)
        lsp_scores = torch.tensor(lsp_scores, dtype=torch.float, device=self.device)
        # print(lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores)
        return list(lsp_decoding_ids), list(lsp_past_key_values), lsp_probs, lsp_scores

    
    def _update_bank(self, bank, best_decoding_ids, best_past_key_values, best_probs, best_scores):
        # best_generations = [self.generator.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True) for decoding_ids in best_decoding_ids]
        if bank is None:
            bank = [best_decoding_ids, best_past_key_values, best_probs, best_scores]
            # bank.append((best_decoding_ids, best_past_key_values, best_probs, best_scores, best_generations))
        else:
            bank[0].extend(best_decoding_ids)
            bank[1].extend(best_past_key_values)
            bank[2] = torch.cat([bank[2], best_probs], 0)
            bank[3] = torch.cat([bank[3], best_scores], 0)
            # bank[4].extend(best_generations)
        return bank

    def _select_from_bank(self, bank, beam_size=2, verbose=False):
        cand_decoding_ids, cand_past_key_values, cand_probs, cand_scores = bank
        if verbose:
            logging.info(f"[Bank] {len(cand_decoding_ids)} candidates")
            funcs_with_scores = []
            for idx in range(len(cand_decoding_ids)):
                func = self.generator.tokenizer.decode(cand_decoding_ids[idx], skip_special_tokens=True)
                score = cand_scores[idx].item()
                funcs_with_scores.append((func, score))
            for func, score in sorted(funcs_with_scores, key=lambda p: p[-1], reverse=True):
                logging.info(f"\tfunc: ```{func}```")
                logging.info(f"\t\tscore: {score}")
        beam_scores, best_idx = cand_scores.topk(beam_size, 0, True, True)
        beam_probs = cand_probs[best_idx]
        beam_decoding_ids = [cand_decoding_ids[best_idx[i].item()] for i in range(beam_size)]
        beam_past_key_values = [cand_past_key_values[best_idx[i].item()] for i in range(beam_size)]
        return beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores

    def generate_with_lsp(
            self,
            desc,
            file_path,
            code_context,
            line,
            column,
            max_len=256,
            beam_size=2,
            cand_num=3,
            lsp_threshold=0.8,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6,
            repetition_penalty=1.0,
        ):
        source_ids = self.generator.tokenizer([desc], add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.device)
        attention_mask = source_ids.ne(self.generator.tokenizer.pad_token_id)

        logits_processor = LogitsProcessorList()
        if repetition_penalty > 1:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        encoder_outputs, beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores = self._init_decoding(source_ids, attention_mask, beam_size)

        final_generation_ids = []
        for step_idx in range(1, max_len):
            for beam_idx, decoding_ids in enumerate(beam_decoding_ids):
                if sum(decoding_ids.eq(self.generator.tokenizer.eos_token_id).long()) >= 1: # type: ignore
                    # func = self.generator.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True).strip()
                    # if len(func) == 0:
                    #     continue
                    if sum(decoding_ids.eq(self.generator.tokenizer.bos_token_id).long()) >= 1:
                        final_generation_ids.append((decoding_ids[1:], beam_scores[beam_idx].item()))
                    beam_scores[beam_idx] = -1e10
            if len(final_generation_ids) >= beam_size:
                break
            expanded_beam_decoding_ids, expanded_beam_past_key_values, expanded_beam_probs, expanded_beam_scores = [], [], [], []
            expanded = False
            for beam_idx in range(len(beam_decoding_ids)):
                func = self.generator.tokenizer.decode(beam_decoding_ids[beam_idx][1:], skip_special_tokens=True)
                if not func.strip().endswith(PLM_LSP_POINT) or beam_probs[beam_idx].item() < lsp_threshold:
                    expanded_beam_decoding_ids.append(beam_decoding_ids[beam_idx])
                    expanded_beam_past_key_values.append(beam_past_key_values[beam_idx])
                    expanded_beam_probs.append(beam_probs[beam_idx:beam_idx + 1])
                    expanded_beam_scores.append(beam_scores[beam_idx:beam_idx + 1])
                    continue
                func = clean_lsp(func)
                logging.info(f"[ENTER LSP] confidence: {beam_probs[beam_idx].item()} >= {lsp_threshold}")
                logging.info(f"uncomplete func: ```{repr(func)}```")
                logging.info(f"current score: {beam_scores[beam_idx].item()}")
                # logging.info(f"\tcurrent code: ```\n{func}\n```")
                cands = self.get_lsp_completions(func, file_path, code_context, line, column)
                logging.info(f"\tcandidates by LSP: {cands}")
                if len(cands) == 0:
                    expanded_beam_decoding_ids.append(beam_decoding_ids[beam_idx])
                    expanded_beam_past_key_values.append(beam_past_key_values[beam_idx])
                    expanded_beam_probs.append(beam_probs[beam_idx:beam_idx + 1])
                    expanded_beam_scores.append(beam_scores[beam_idx:beam_idx + 1])
                    continue

                lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores = self._lsp_expand(
                    cands,
                    encoder_outputs,
                    attention_mask,
                    beam_decoding_ids[beam_idx],
                    beam_past_key_values[beam_idx],
                    logits_processor,
                    cand_num=cand_num,
                    token_threshold=token_threshold,
                    token_k=token_k,
                    temperature=temperature
                )
                if len(lsp_decoding_ids) == 0:
                    expanded_beam_decoding_ids.append(beam_decoding_ids[beam_idx])
                    expanded_beam_past_key_values.append(beam_past_key_values[beam_idx])
                    expanded_beam_probs.append(beam_probs[beam_idx:beam_idx + 1])
                    expanded_beam_scores.append(beam_scores[beam_idx:beam_idx + 1])
                    continue
                
                expanded = True
                # lsp_scores = beam_scores[beam_idx:beam_idx+1].repeat(len(lsp_decoding_ids))
                lsp_scores = beam_scores[beam_idx:beam_idx+1].repeat(len(lsp_decoding_ids)) + lsp_scores

                # expanded_beam_decoding_ids.append(beam_decoding_ids[beam_idx])
                # expanded_beam_past_key_values.append(beam_past_key_values[beam_idx])
                # expanded_beam_probs.append(beam_probs[beam_idx:beam_idx + 1])
                # expanded_beam_scores.append(beam_scores[beam_idx:beam_idx + 1])
                expanded_beam_decoding_ids.extend(lsp_decoding_ids)
                expanded_beam_past_key_values.extend(lsp_past_key_values)
                expanded_beam_probs.append(lsp_probs)
                expanded_beam_scores.append(lsp_scores)

            beam_decoding_ids = expanded_beam_decoding_ids
            beam_past_key_values = expanded_beam_past_key_values
            beam_probs = torch.cat(expanded_beam_probs, 0)
            beam_scores = torch.cat(expanded_beam_scores, 0)

            bank = None
            for beam_idx in range(len(beam_decoding_ids)):
                best_decoding_ids, best_past_key_values, best_probs, best_scores = self._beam_advance(
                    encoder_outputs,
                    attention_mask,
                    beam_decoding_ids[beam_idx],
                    beam_past_key_values[beam_idx],
                    beam_scores[beam_idx],
                    logits_processor,
                    beam_size
                )
                bank = self._update_bank(bank, best_decoding_ids, best_past_key_values, best_probs, best_scores)
            
            if not bank:
                break
            beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores = self._select_from_bank(bank, beam_size, verbose=expanded)

        # for l, (tmp_generated_ids, tmp_decoder_input_ids, tmp_past_key_values, tmp_scores) in bank.items():
        #     for ids, s in zip(tmp_generated_ids, tmp_scores):
        #         func = self.generator.tokenizer.decode(ids[1:], skip_special_tokens=True)
        #         cands.append((func, s))
        if len(final_generation_ids) == 0:
            for i, decoding_ids in enumerate(beam_decoding_ids): 
                # func, score = self.generator.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True).strip(), beam_scores[i].item()
                # if len(func) == 0:
                #     continue
                final_generation_ids.append((decoding_ids[1:], beam_scores[i].item()))
        
        final_generation_ids = list(sorted(final_generation_ids, key=lambda pair:pair[-1], reverse=True))
        final_generations = []
        logging.info(f"[PREDICTIONS]")
        for idx, (ids, s) in enumerate(final_generation_ids, 1):
            logging.info(f"Prediction {idx}, socre: {s}")
            logging.info(f"\n{self.generator.tokenizer.decode(ids, skip_special_tokens=True).strip()}")
            func = self.generator.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if len(func) == 0:
                continue
            final_generations.append((func, s))
        logging.info("")
        return final_generations[0][0]
    
    def update_lsp_project(self, pj_path, py_env=None):
        if self.jedi_pj is None or self.jedi_pj._path != pj_path:
            self.jedi_pj: jedi.Project = jedi.Project(pj_path, environment_path=py_env, added_sys_path=(pj_path,))

    def get_lsp_context(self, file_path: str, func_code:str):
        with Path(file_path).open("r") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except Exception as e:
            return None, None, None
        funcs = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
        
        for func in funcs:
            func_source = ast.get_source_segment(code, func)
            line, column = func.lineno, func.col_offset
            rest_source = code.replace(func_source, "<PLACEHOLDER>")
            _, tokens = DataAugmentor.tokenize_func(func_source)
            original_func = "".join(tokens).replace(PLM_LSP_POINT, "<unk>").strip()
            if original_func == func_code.replace(PLM_LSP_POINT, "").strip():
                return rest_source, line, column
        return None, None, None
    
    def get_lsp_completions(self, func, file_path, code_context, line, column):
        lines = func.split("\n")
        _line, _column = line, column
        if len(lines) == 1:
            _column += len(lines[0])
        else:
            _line += len(lines) - 1
            _column = len(lines[-1])
        context = code_context.replace("<PLACEHOLDER>", func)
        # logging.info(f"context: ```\n{context}\n```")
        try:
            script = jedi.Script(project=self.jedi_pj, path=file_path, code=context)
            completions = script.complete(line=_line, column=_column)
        except Exception as e:
            completions = []
        completions = [completion.complete.strip() for completion in completions]
        completions = [
            completion for completion in completions if len(completion) > 0 and completion not in SKIPPED
        ]
        return completions