import theano
import theano.tensor as T
import numpy as np
import logging
import copy

from model import Model
from components import CondAttLSTM

from config import config_info
import config
from lang.grammar import Grammar
from parse import *
from astnode import *
from util import is_numeric
from components import Hyp, PointerNet, CondAttLSTM
from nn.utils.theano_utils import *

from retrieval import NGramSearcher


class CondAttLSTMAligner(CondAttLSTM):
    def __init__(self, *args, **kwargs):

        super(CondAttLSTMAligner, self).__init__(*args, **kwargs)

    def _step_align(self,
                    t, xi_t, xf_t, xo_t, xc_t, mask_t, parent_t,
                    h_tm1, c_tm1, hist_h,
                    u_i, u_f, u_o, u_c,
                    c_i, c_f, c_o, c_c,
                    h_i, h_f, h_o, h_c,
                    p_i, p_f, p_o, p_c,
                    att_h_w1, att_w2, att_b2,
                    context, context_mask, context_att_trans,
                    b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, att_h_w1)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])
        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, att_w2) + att_b2
        att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw - T.max(att_raw, axis=-1, keepdims=True))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)
        # (batch_size, context_dim)
        scores = ctx_att[:, :, None]
        ctx_vec = T.sum(context * scores, axis=1)
        ##### attention over history #####

        def _attention_over_history():
            hist_h_mask = T.zeros((hist_h.shape[0], hist_h.shape[1]), dtype='int8')
            hist_h_mask = T.set_subtensor(hist_h_mask[:, T.arange(t)], 1)

            hist_h_att_trans = T.dot(hist_h, self.hatt_hist_W1) + self.hatt_b1
            h_tm1_hatt_trans = T.dot(h_tm1, self.hatt_h_W1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))
            hatt_exp = T.exp(hatt_raw - T.max(hatt_raw, axis=-1, keepdims=True)) * hist_h_mask
            h_att_weights = hatt_exp / (T.sum(hatt_exp, axis=-1, keepdims=True) + 1e-7)

            # (batch_size, output_dim)
            _h_ctx_vec = T.sum(hist_h * scores, axis=1)

            return _h_ctx_vec

        h_ctx_vec = T.switch(t,
                             _attention_over_history(),
                             T.zeros_like(h_tm1))

        if not config.parent_hidden_state_feed:
            t = 0

        par_h = T.switch(t,
                         hist_h[T.arange(hist_h.shape[0]), parent_t, :],
                         T.zeros_like(h_tm1))

        ##### feed in parent hidden state #####
        if config.tree_attention:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i) + T.dot(h_ctx_vec, h_i))
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f) + T.dot(h_ctx_vec, h_f))
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c) + T.dot(h_ctx_vec, h_c))
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o) + T.dot(h_ctx_vec, h_o))
        else:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i))  # + T.dot(h_ctx_vec, h_i)
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f))  # + T.dot(h_ctx_vec, h_f)
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c))  # + T.dot(h_ctx_vec, h_c)
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o))  # + T.dot(h_ctx_vec, h_o)
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        new_hist_h = T.set_subtensor(hist_h[:, t, :], h_t)

        return h_t, c_t, scores, new_hist_h

    def align(self, X, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None,
              mask=None, context_mask=None, srng=None, time_steps=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        # (n_timestep, batch_size)
        mask = self.get_mask(mask, X)
        # (n_timestep, batch_size, input_dim)
        X = X.dimshuffle((1, 0, 2))

        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)

        # (n_timestep, batch_size, output_dim)
        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = T.dot(context, self.att_ctx_W1) + self.att_b1

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_cell:
            # (batch_size, output_dim)
            first_cell = T.unbroadcast(init_cell, 1)
        else:
            first_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if not hist_h:
            # (batch_size, n_timestep, output_dim)
            hist_h = alloc_zeros_matrix(X.shape[1], X.shape[0], self.output_dim)

        n_timestep = X.shape[0]
        time_steps = T.arange(n_timestep, dtype='int32')

        # (n_timestep, batch_size)
        parent_t_seq = parent_t_seq.dimshuffle((1, 0))

        [outputs, cells, att_scores, hist_h_outputs], updates = theano.scan(
            self._step_align,
            sequences=[time_steps, xi, xf, xo, xc, mask, parent_t_seq],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell
                None,
                hist_h,  # for hist_h
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.H_i, self.H_f, self.H_o, self.H_c,
                self.P_i, self.P_f, self.P_o, self.P_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        att_scores = att_scores.dimshuffle((1, 0, 2))

        return att_scores


class RetrievalModel(Model):
    def __init__(self, regular_model=None):
        """
        super(RetrievalModel, self).__init__()
        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        """
        super(RetrievalModel, self).__init__()

        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        # update params for new decoder
        self.params = self.query_embedding.params + self.query_encoder_lstm.params + \
            self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
            [self.rule_embedding_W, self.rule_embedding_b, self.node_embedding, self.vocab_embedding_W, self.vocab_embedding_b] + \
            self.decoder_hidden_state_W_rule.params + self.decoder_hidden_state_W_token.params

    def build(self):
        super(RetrievalModel, self).build()
        self.build_aligner()

    def build_aligner(self):
        tgt_action_seq = ndim_itensor(3, 'tgt_action_seq')
        tgt_action_seq_type = ndim_itensor(3, 'tgt_action_seq_type')
        tgt_node_seq = ndim_itensor(2, 'tgt_node_seq')
        tgt_par_rule_seq = ndim_itensor(2, 'tgt_par_rule_seq')
        tgt_par_t_seq = ndim_itensor(2, 'tgt_par_t_seq')

        tgt_node_embed = self.node_embedding[tgt_node_seq]
        query_tokens = ndim_itensor(2, 'query_tokens')
        query_token_embed, query_token_embed_mask = self.query_embedding(
            query_tokens, mask_zero=True)
        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]

        tgt_action_seq_embed = T.switch(T.shape_padright(tgt_action_seq[:, :, 0] > 0),
                                        self.rule_embedding_W[tgt_action_seq[:, :, 0]],
                                        self.vocab_embedding_W[tgt_action_seq[:, :, 1]])
        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)
        tgt_par_rule_embed = T.switch(tgt_par_rule_seq[:, :, None] < 0,
                                      T.alloc(0., 1, config.rule_embed_dim),
                                      self.rule_embedding_W[tgt_par_rule_seq])

        if not config.frontier_node_type_feed:
            tgt_node_embed *= 0.
        if not config.parent_action_feed:
            tgt_par_rule_embed *= 0.

        decoder_input = T.concatenate(
            [tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed], axis=-1)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask, dropout=0, srng=self.srng)

        tgt_action_seq_mask = T.any(tgt_action_seq_type, axis=-1)

        alignments = self.decoder_lstm.align(decoder_input, context=query_embed,
                                             context_mask=query_token_embed_mask,
                                             mask=tgt_action_seq_mask,
                                             parent_t_seq=tgt_par_t_seq,
                                             srng=self.srng)

        alignment_inputs = [query_tokens, tgt_action_seq, tgt_action_seq_type,
                            tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq]
        self.align = theano.function(alignment_inputs, [alignments])

    def decode_with_retrieval(self, example, grammar, terminal_vocab, ngram_searcher, beam_size, max_time_step, log=False):
        # beam search decoding with ngram retrieval
        eos = terminal_vocab.eos
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W.get_value(borrow=True)
        rule_embedding = self.rule_embedding_W.get_value(borrow=True)

        query_tokens = example.data[0]
        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)
        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp_ng(grammar)
        root_hyp.state = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.cell = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.action_embed = np.zeros(config.rule_embed_dim).astype('float32')
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query][:config.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else:
                token_set.add(tid)

        for t in xrange(max_time_step):
            hyp_num = len(hyp_samples)
            decoder_prev_state = np.array([hyp.state for hyp in hyp_samples]).astype('float32')
            decoder_prev_cell = np.array([hyp.cell for hyp in hyp_samples]).astype('float32')
            hist_h = np.zeros((hyp_num, max_time_step, config.decoder_hidden_dim)).astype('float32')

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h

            prev_action_embed = np.array(
                [hyp.action_embed for hyp in hyp_samples]).astype('float32')
            node_id = np.array([hyp.node_id for hyp in hyp_samples], dtype='int32')
            parent_rule_id = np.array([hyp.parent_rule_id for hyp in hyp_samples], dtype='int32')
            parent_t = np.array([hyp.get_action_parent_t() for hyp in hyp_samples], dtype='int32')
            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [np.array([t], dtype='int32'), decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, \
                rule_prob, gen_action_prob, vocab_prob, copy_prob = self.decoder_func_next_step(
                    *inputs)

            rule_prob, vocab_prob, copy_prob = update_probs(
                rule_prob, vocab_prob, copy_prob, hyp_samples, ngram_searcher, grammar=grammar)

            new_hyp_samples = []
            cut_off_k = beam_size
            score_heap = []

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.as_type_node] if config.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]
                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score
                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)
                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids,
                                                  None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for k, cand_id in enumerate(top_cand_ids):
                # cand is rule application
                # verbose = k==0
                verbose = False
                new_hyp = None
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp_ng(hyp)
                    new_hyp.apply_rule(rule)
                    new_hyp.to_move = False
                    new_hyp.update_ngrams(ngram_searcher.get_keys(
                        hyp.get_ngrams(), rule_id, "APPLY_RULE", verbose))
                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]
                    new_hyp = Hyp_ng(hyp)
                    new_hyp.append_token(token)

                    if tid == unk:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(hyp.get_ngrams(),
                                                                      unk_pos, "COPY_TOKEN", verbose))
                    elif tid in src_token_id:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(hyp.get_ngrams(),
                                                                      src_token_id.index(tid), "COPY_TOKEN", verbose))
                    else:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(
                            hyp.get_ngrams(), tid, "GEN_TOKEN", verbose))

                    # look at parent timestep ?
                    if tid == eos:
                        new_hyp.to_move = True
                    else:
                        new_hyp.to_move = False

                    if log:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]
                        if cand_copy_prob > 0.5:
                            new_hyp.log += ' || ' + \
                                str(new_hyp.frontier_nt()) + \
                                '{copy[%s][p=%f]}' % (token, cand_copy_prob)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)

                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]
                    new_hyp_samples.append(new_hyp)

                # cand is word generation
            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps


class Hyp_ng(Hyp):
    def __init__(self, *args):
        super(Hyp_ng, self).__init__(*args)
        if isinstance(args[0], Hyp):
            self.hist_ng = copy.copy(args[0].hist_ng)
        else:
            self.hist_ng = []
        self.to_move = False

    def update_ngrams(self, new_ngram):
        # print new_ngram
        self.hist_ng.append(new_ngram)

    def get_ngrams(self, verbose=False):

        try:
            if self.to_move:
                t = self.get_action_parent_t()
            else:
                t = self.t

            k = self.hist_ng[t]
            if verbose:
                print self.to_move
                print t
                print len(self.hist_ng)
                print self.tree.pretty_print()
            return k
        except:
            return [None for i in range(config.max_ngrams+1)]


def update_probs(rule_prob, vocab_prob, copy_prob, hyp_samples, ngram_searcher, grammar=None):
    f = config.retrieval_factor
    # print f, type(f)

    for k, hyp in enumerate(hyp_samples):
        verbose = False
        # if k == 0:
        #     verbose = True
        ngram_keys = hyp.get_ngrams(verbose)
        # if k == 0:
        #    print ngram_keys
        for value, score, flag in ngram_searcher(ngram_keys):

            if flag == "APPLY_RULE":
                # if grammar is not None and k == 0:
                #     print "candidate rule :"
                #     print grammar.rules[value]
                rule_prob[k, value] *= np.exp(f*score)
                
                #print("---- apply rule here ----")
            elif flag == "GEN_TOKEN":
                vocab_prob[k, value] *= np.exp(f*score)
                
            else:
                assert flag == "COPY_TOKEN"
                if value < config.max_query_length:
                    copy_prob[k, value] *= np.exp(f*score)

        rule_prob[k] /= rule_prob[k].sum()
        vocab_prob[k] /= vocab_prob[k].sum()
        copy_prob[k] /= copy_prob[k].sum()

    return rule_prob, vocab_prob, copy_prob
