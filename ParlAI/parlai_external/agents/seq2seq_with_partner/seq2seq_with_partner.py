from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.agents.seq2seq.modules import Seq2seq, opt_to_kwargs
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.core.utils import NEAR_INF, padded_tensor, round_sigfigs, warn_once, AttrDict
from parlai_external.agents.seq2seq_with_partner.beam import Beam
import numpy as np

from parlai_external.agents.seq2seq_with_partner.paths import paths as dump_paths
import json
import copy
import math
import time
import os

# l1 : list of lists, of length L
# l2 : list of lists, of length L
# output : list of lists, of length L
def join_1dlist(l1, l2):
    return [i1+i2 for i1, i2 in zip(l1, l2)]

# l1 : list of lists of lists, of length L
# l2 : list of lists of lists, of length L
# output : list of lists, of length L
def join_2dlist(L1, L2):
    bsz = len(L1)
    bwidth = len(L1[0])
    return [[L1[batch_idx][beam_idx] + [ L2[batch_idx][beam_idx] ] for beam_idx in range(bwidth)] for batch_idx in range(bsz)]

class Output(AttrDict):
    """
    Output is a namedtuple containing agent predictions.

    This is the expected return type of the train_step and eval_step functions,
    though agents can choose to return None if they do not want to answer.

    .. py:attribute:: text

        list of strings of length bsz containing the predictions of the model

    .. py:attribute:: text_candidates

        list of lists of length bsz containing ranked predictions of the model.
        each sub-list is an ordered ranking of strings, of variable length.
    """

    def __init__(self, text=None, text_candidates=None, mtbeam_log=None, **kwargs):
        super().__init__(text=text, text_candidates=text_candidates, mtbeam_log=mtbeam_log, **kwargs)

class Seq2seqPartnerAgent(Seq2seqAgent):
    """
    This agent inherits from Seq2seqAgent and supports adding
    *already trained* partner model for running multi-turn beam search
    """

    def predscore_dic(self, pred, score, idx, iteration, predecessor):
        return {"id":idx,
                "iter":iteration,
                "predecessor":predecessor,
                "pred":self._v2t(pred),
                "score":round_sigfigs(score.item(),4),
               }

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        Seq2seqAgent.add_cmdline_args(argparser)

        agent = argparser.add_argument_group('Seq2SeqPartner Arguments')
        agent.add_argument('--init-partner-model', required=True, type=str,
                           help='load partner dict/model/opts from this path')
        agent.add_argument('--mtbeam', type=int, default=0,
                           help='Number of iterations in mtbeam search')
        agent.add_argument("--egocentric", default=False, action="store_true",
                            help="Use CSM as PSM (egocentric or mindless)")
        agent.add_argument('--model-str', type=str, default="",
                           help='model string for saving results')
        agent.add_argument("--write", default=False, action="store_true")
        agent.add_argument('--beam-num-iterations', type=int, default=1, hidden=True,
                           help='Number of iterations for iterbeam')
        agent.add_argument('--beam-dist-threshold', type=int, default=1, hidden=True,
                           help='Distance threshold for iterbeam')
        return agent

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions.
        Version 1 split from version 0 on Aug 29, 2018.
        Version 2 split from version 1 on Nov 13, 2018
        To use version 0, use --model legacy:seq2seq:0
        To use version 1, use --model legacy:seq2seq:1
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self._cnt = 0
        self.dev = 'cpu' if opt.get('no_cuda') else 'cuda'
        self.id = 'Seq2SeqPartner'
        self.init_partner_model_path = opt.get('init_partner_model', None)
        self.metrics['rate'] = 0
        self.metrics['good_batch'] = 0
        self.metrics['all_batch'] = 0
        self.beam_num_iterations = opt.get('beam_num_iterations', 1)
        self.beam_dist_threshold = opt.get('beam_dist_threshold', 1)
        self.write = opt.get('write', False)
        self.bwidth = self.beam_num_iterations if self.beam_num_iterations != 1 else self.beam_size
        self.expand_beam = opt.get('expand_beam')

        self.extra_args = {\
            "start":self.START_IDX,
            "end":self.END_IDX,
            "pad":self.NULL_IDX,
            "min_length":self.beam_min_length,
            "min_n_best":self.beam_size, # NOTE min_n_best = beam_size
            "beam_size":self.beam_size,
            "block_ngram":self.beam_block_ngram,
            "beam_num_iterations": self.beam_num_iterations,
            "iterbeam_distance": self.beam_dist_threshold,
        }

        if shared:
            self.partner_model = shared['partner_model']
            partner_states = shared.get('partner_states', {})
            self.num_turns = shared['mtbeam']
        else:
            self.num_turns = opt.get('mtbeam', 0) + 1
            self.partner_model = copy.deepcopy(self.model)
            partner_states = torch.load(self.init_partner_model_path, map_location=lambda cpu, _: cpu)
            assert 'model' in partner_states, 'Partner model states does not have model field'
            self.partner_model.load_state_dict(partner_states['model'])
            self.partner_model.decoder.rnn.flatten_parameters()
            self.partner_model.encoder.rnn.flatten_parameters()
        self.tom = opt.get('egocentric')
        self.bsz = opt.get('batchsize')
        self.model_str = "{}.iter{}.bw{}.mt{}.th{}".format(self.dev,
                self.beam_num_iterations, self.beam_size, self.num_turns, self.beam_dist_threshold)\
                if opt.get('model_str') == "" else opt.get('model_str')

        persona_indicator = ['your', 'persona', ':']
        self.persona_indicator_inds = []
        for tok in persona_indicator:
            ind = self.dict.tok2ind[tok]
            self.persona_indicator_inds.append(ind)

        if self.write:
            root_path = dump_paths[os.environ.get('USER')]
            self.hyp = open(root_path+"/hypo_dump/hyp_"+self.model_str, "w")
            self.json_out = {}

    def share(self):
        shared = super().share()
        shared['partner_model'] = self.partner_model
        shared['partner_states'] = None # we dont need it
        shared['mtbeam'] = self.num_turns

        return shared

    def train_step(self, batch):
        """Train on a single batch of examples.
        """
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(self.model, self.criterion, batchsize,
                               self.truncate or 180)
        self.model.train()
        self.zero_grad()

        try:
            scores, preds, _ = self.model(batch.text_vec, batch.label_vec)
            score_view = scores.view(-1, scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            loss /= target_tokens  # average loss per token
            loss.backward()
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
            else:
                raise e

    def no_context(self, history_vec):
        if history_vec[:3].tolist() != self.persona_indicator_inds:
            return True
        else:
            return False

    def maybe_remove_context(self, memory_vecs, use_context=True):
        history = []
        for example in memory_vecs:
            new_example = example if use_context else filter(self.no_context, example)
            history.append( [history_vec.to(self.dev) for history_vec in new_example] )
        return history

    def take_real_text_vec(self, text_vec, text_lengths):
        return [ [ text_vec[ii][:text_lengths[ii]] ] for ii in range(len(text_vec))]

    def cat_history(self, history_list):
        newline = torch.Tensor([self.dict.tok2ind['\n']]).long().to(self.dev)
        merged = [ [newline] * (len(history)*2-1) for history in history_list]
        for idx in range(len(history_list)):
            merged[idx][0::2] = history_list[idx]
        merged = [torch.cat(item, 0) for item in merged]
        numels = [item.numel() for item in merged]
        maxlen = max(numels)
        merged = [torch.cat([item, item.new_zeros(maxlen-numel)], 0) for item, numel in zip(merged, numels)]
        merged = torch.stack(merged, 0)

        return merged, numels

    def ipdb_if(self):
        if self._cnt == 223:
            import ipdb; ipdb.set_trace()

    def eval_step(self, batch):
        with torch.no_grad():
            return self.eval_step_(batch)

    def eval_step_(self, batch):
        """Evaluate a single batch of examples.
        Here we use split_lines in vectorize, so we need to take care of memory_vecs
        """
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        bwidth = self.bwidth
        self.model.eval()
        self.partner_model.eval()
        cand_scores = None
        cand_choices = None

        persona_memvecs = self.maybe_remove_context(batch.memory_vecs, use_context=True) # bsz x 4
        nopersona_memvecs = self.maybe_remove_context(batch.memory_vecs, use_context=False) # bsz x k
        real_text_vec = self.take_real_text_vec(batch.text_vec, batch.text_lengths) # bsz x 1+
        all_history = join_1dlist(persona_memvecs, real_text_vec) # bsz x 5+
        input_vec, text_lengths = self.cat_history(all_history) # bsz x tensor
        batch_next = Batch(text_vec=input_vec, text_lengths=text_lengths)
        context = self._v2t(input_vec[0])
        mtbeam_log = {}
        mtbeam_log['context'] = context

        self._cnt += 1
        if self.num_turns == 1: # Single-Turn Beam Search
            if self.skip_generation:
                warn_once(
                    "--skip-generation does not produce accurate metrics beyond ppl",
                    RuntimeWarning
                )
                logits, preds, _ = self.model(batch.text_vec, batch.label_vec)
            elif self.beam_size == 1:
                # greedy decode
                logits, preds, _ = self.model(batch.text_vec)
            elif self.beam_size > 1:
                out = self.beam_search(
                    self.model,
                    batch_next,
                    text_lengths,
                    **self.extra_args
                )
                beam_preds_scores, n_best_preds_scores, beams = out
                preds = [batch_e[0][0] for batch_e in beam_preds_scores]

                if self.beam_dot_log is True:
                    self._write_beam_dots(batch.text_vec, beams)
                csm_best_preds = out[0]
                csm_dics = [self.predscore_dic(*pred_score, idx + 1, 0, 0) \
                            for idx, pred_score in enumerate(csm_best_preds[0])]
                mtbeam_log[0] = csm_dics
                mtbeam_log['final'] = 0
                mtbeam_log['best'] = 0

                if self.write:
                    self.json_out[self._cnt] = mtbeam_log
                    if self._cnt % 1 == 0:
                        root_path = dump_paths[os.environ.get('USER')]
                        self.json_dest = open(root_path+"/json_dump/"+self.model_str+'.json', "w")
                        json.dump(self.json_out, self.json_dest)
                        self.json_dest.close()

        else: # Multi-turn Beam Search
            big_persona_memvecs = [item for item in persona_memvecs for idx in range(bwidth)] # bsz*bwidth x 4
            big_nopersona_memvecs = [item for item in nopersona_memvecs for idx in range(bwidth)] # bsz*bwidth x k
            csm_best_preds, _, _ = self.beam_search(self.model, batch_next, text_lengths,
                **self.extra_args) # (bsz x bwidth x (preds, scores))
            text_vec = [ [ real_text_vec[batch_idx] + [ beam_e[0] ] for beam_idx, beam_e in enumerate(batch_e) ] \
                for batch_idx, batch_e in enumerate(csm_best_preds) ] # (bsz x bwidth x [list_of_utterances] )
            # text_vec keeps track of the previous utterances
            # so far, contains the previous utterance and the output from the first CSM model
            scores_acc = torch.stack([ torch.stack([ beam_e[1] for beam_e in batch_e ], 0) \
                        for batch_e in csm_best_preds ], 0) # (bsz x bwidth)
            csm_dics = [self.predscore_dic(*pred_score, idx + 1, 0, 0) for idx, pred_score in enumerate(csm_best_preds[0])]
            mtbeam_log[0] = csm_dics

            backtrack_ids = None
            for step_idx in range(self.num_turns - 1):
                model = self.partner_model if step_idx % 2 == 0 and not self.tom else self.model
                memvecs = big_nopersona_memvecs if step_idx % 2 == 0 and not self.tom else big_persona_memvecs # bsz x 4
                all_history = join_1dlist(memvecs,
                                            [ beam_e for batch_e in text_vec for beam_e in batch_e ] )
                # bsz*bwidth x 4+
                text_vec_next, text_lengths = self.cat_history(all_history)
                batch_next = Batch(text_vec=text_vec_next, text_lengths=text_lengths)
                cur_scores_preds, _, _ = self.beam_search(model, batch_next, text_lengths, **self.extra_args)
                # bsz*bwidth x bwidth x (preds, scores)

                psm_dics = [self.predscore_dic(*pred_score, beam_idx1 * bwidth + beam_idx2, step_idx+1, beam_idx1)\
                            for beam_idx1, same_beam_scores_preds in enumerate(cur_scores_preds) \
                           for beam_idx2, pred_score in enumerate(same_beam_scores_preds)]

                cur_scores = torch.Tensor([[beam_e[1] for beam_e in batch_e] \
                    for batch_e in cur_scores_preds]).to(self.dev).view(bsz, -1) # bsz x bwidth^^2
                csm_scores = scores_acc.view(-1)[:,None].repeat(1, bwidth).view(bsz, -1)
                combined_scores = csm_scores + cur_scores # bsz x bwidth^^2
                top_scores, top_ids = torch.topk(combined_scores, bwidth, dim=1)
                # bsz x bwidth
                ids = top_ids / bwidth
                backtrack_ids = backtrack_ids.gather(1, ids) if not backtrack_ids is None else ids
                mtbeam_log[step_idx+1] = ( psm_dics, top_ids.cpu().numpy().tolist() )

                if self.num_turns > 2:
                    rem_ids = torch.remainder(top_ids, bwidth)
                    best_prev_hyp = [ [ text_vec[batch_idx][beam_e] for beam_e in batch_e ] \
                                     for batch_idx, batch_e in enumerate(backtrack_ids.cpu().numpy().tolist()) ]
                    # bsz x bwidth x [list of prev utterances]
                    best_cur_hyp = [ [ cur_scores_preds[batch_idx*bwidth + beam_e//bwidth][beam_e % bwidth][0] for beam_e in batch_e ] \
                                    for batch_idx, batch_e in enumerate(top_ids.cpu().numpy().tolist()) ]
                    # bsz x bwidth x Tensor (utterance)
                    text_vec = join_2dlist(best_prev_hyp, best_cur_hyp)
                    scores_acc = top_scores

            best_idx_after_psm = backtrack_ids[:,0] # bsz
            self.metrics['all_batch'] += bsz
            chg_idx = (best_idx_after_psm != 0)
            nochg_idx = (best_idx_after_psm == 0)
            self.metrics['good_batch'] += chg_idx.sum().item()
            preds = [csm_best_preds[batch_idx][beam_idx][0] for batch_idx, beam_idx in enumerate(best_idx_after_psm.cpu().numpy().tolist())]

            mtbeam_log['final'] = top_ids[0][0].item()
            mtbeam_log['best'] = best_idx_after_psm[0].item()

            if self.write:
                self.json_out[self._cnt] = mtbeam_log

                if self._cnt % 1 == 0:
                    root_path = dump_paths[os.environ.get('USER')]
                    self.json_dest = open(root_path+"/json_dump/"+self.model_str+'.json', "w")
                    json.dump(self.json_out, self.json_dest)
                    self.json_dest.close()

        #if batch.label_vec is not None:
        if False:
            # calculate loss on targets with teacher forcing
            # we need to make new text_vec here since we do split memvecs
            persona_memvecs = self.maybe_remove_context(batch.memory_vecs, use_context=True)
            real_text_vec = self.take_real_text_vec(batch.text_vec, batch.text_lengths)
            all_history = join_1dlist(persona_memvecs, real_text_vec)
            input_vec, text_lengths = self.cat_history(all_history)
            f_scores, f_preds, _ = self.model(input_vec, batch.label_vec)
            score_view = f_scores.view(-1, f_scores.size(-1))
            loss = self.criterion(score_view, batch.label_vec.view(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == f_preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(batch.text_vec)
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds]
        if self.write:
            self.hyp.write("\n".join( text ))
            self.hyp.write("\n")
            self.hyp.flush()
        return Output(text=text, text_candidates=cand_choices, mtbeam_log=mtbeam_log)

    def observe(self, observation):
        """Process incoming message in preparation for producing a response.
        This includes remembering the past history of the conversation.
        """
        reply = self.last_reply(
            use_label=(self.opt.get('use_reply', 'label') == 'label'))
        self.observation = self.get_dialog_history(
            observation, reply=reply, add_person_tokens=self.add_person_tokens,
            add_p1_after_newln=self.opt.get('add_p1_after_newln', False))
        return self.vectorize(self.observation, split_lines=True) # NOTE removed truncate

    def report(self):
        """Report loss and perplexity from model's perspective.
        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        if self.num_turns > 1:
            if self.metrics['all_batch'] == 0 :
                base['rate'] = 0
            else:
                base['rate'] = round_sigfigs(self.metrics['good_batch'] / self.metrics['all_batch'], 4)
        return base


    def _beam_search(self, model, batch, beam_size, beam_num_iterations , start=1, end=2,
                    pad=0, min_length=3, min_n_best=5, max_ts=40, block_ngram=0,
                    iterbeam_distance=1):
        """Beam search given the model and Batch


        This function expects to be given a TorchGeneratorModel. Please refer to
        that interface for information.

        :param TorchGeneratorModel model: Implements the above interface
        :param Batch batch: Batch structure with input and labels
        :param int beam_size: Size of each beam during the search
        :param int start: start of sequence token
        :param int end: end of sequence token
        :param int pad: padding token
        :param int min_length: minimum length of the decoded sequence
        :param int min_n_best: minimum number of completed hypothesis generated
            from each beam
        :param int max_ts: the maximum length of the decoded sequence

        :return: tuple (beam_pred_scores, n_best_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - n_best_preds_scores: list of n_best list of tuples (prediction, score)
              for each sample from Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        encoder_states = model.encoder(batch.text_vec)
        dev = batch.text_vec.device

        bsz = len(batch.text_lengths)

        beams = []
        for iter_idx in range(beam_num_iterations):
            beams.append([
                Beam(beam_size, min_length=min_length, padding_token=pad,
                     bos_token=start, eos_token=end, min_n_best=min_n_best,
                     cuda=dev, block_ngram=block_ngram, dist_threshold=self.beam_dist_threshold,
                     voc_size=len(self.dict.ind2tok))
                for i in range(bsz)
            ])

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)

        iterbeam_hypotheses_stack = {i:[] for i in range(bsz)}
        beam_preds_scores = []

        for iter_idx in range(beam_num_iterations):
            beam_preds_scores.append([])
            incr_state = None
            # repeat encoder outputs and decoder inputs
            decoder_input = torch.LongTensor([start]).expand(bsz * beam_size, 1).to(dev)

            for ts in range(max_ts):
                # exit early if needed
                if all((b.done() for b in beams[iter_idx])):
                    break

                score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
                # only need the final hidden state to make the word prediction
                score = score[:, -1:, :]
                score = model.output(score)
                # score contains softmax scores for bsz * beam_size samples
                score = score.view(bsz, beam_size, -1)
                score = F.log_softmax(score, dim=-1)
                for i, b in enumerate(beams[iter_idx]):
                    if not b.done():
                        b.advance(score[i], iterbeam_hypotheses_stack[i])
                incr_state_inds = torch.cat(
                    [beam_size * i +
                        b.get_backtrack_from_current_step() for i, b in enumerate(beams[iter_idx])])
                incr_state = model.reorder_decoder_incremental_state(
                    incr_state, incr_state_inds
                )
                decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
                selection = torch.cat(
                    [b.get_output_from_current_step() for b in beams[iter_idx]]).unsqueeze(-1)
                decoder_input = torch.cat([decoder_input, selection], dim=-1)

            for i, b in enumerate(beams[iter_idx]):
                n_best_tails = b.get_rescored_finished(n_best=min_n_best)
                for bi in range(len(n_best_tails)):
                    _cand = b.get_pretty_hypothesis(b.get_hyp_from_finished(n_best_tails[bi]))
                    iterbeam_hypotheses_stack[i].append(_cand.cpu().numpy())
                    if bi == 0 or beam_num_iterations == 1:
                        beam_preds_scores[-1].append([_cand, n_best_tails[bi].score])

        for iter_idx in range(beam_num_iterations):
            for b in beams[iter_idx]:
                b.check_finished()

        n_best_beam_preds_scores = None

        return beam_preds_scores, n_best_beam_preds_scores, beams

    # 1) sorts batch elements w.r.t length before self._beam_search
    # 2) undoes sorting after self._beam_search
    # 3) sorts each batch element w.r.t NLL
    def beam_search(self, model, batch, text_lengths, *args, **kwargs):
        text_vec = batch.text_vec
        sorted_order = np.argsort(text_lengths)[::-1]
        reverse_order = np.argsort(sorted_order) # bsz
        sorted_order_tensor = text_vec.new_tensor(sorted_order.tolist())
        sorted_text_vec = text_vec.index_select(0, sorted_order_tensor)
        sorted_text_lengths = [text_lengths[ii] for ii in sorted_order]
        batch_ = Batch(text_vec=sorted_text_vec, text_lengths=sorted_text_lengths)
        result = self._beam_search(model, batch_, *args, **kwargs)
        first = result[0]
        if self.beam_num_iterations == 1: # (1 x (bsz x bwidth) x 2)
            real_bsz = len(first[0]) // self.beam_size
            first_t = [first[0][batch_idx*self.beam_size : \
                                (batch_idx+1)*self.beam_size] for batch_idx in range(real_bsz)]
        else: # (bwidth x bsz x 2) 
            first_t = list(map(list, zip(*first)))
        # (bsz x bwidth x (preds, scores))
        first_t = [sorted(first_t[order], key = lambda x:x[1].item(), reverse=True) for order in reverse_order]
        return [first_t, *result[1:]]

