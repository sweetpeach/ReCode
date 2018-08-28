import config
import copy
import simi
from nn.utils.io_utils import deserialize_from_file
from astnode import ASTNode
from dataset import Action

import numpy as np
import re


APPLY_RULE = 0
GEN_TOKEN = 1
COPY_TOKEN = 2
GEN_COPY_TOKEN = 3

ACTION_NAMES = {APPLY_RULE: 'APPLY_RULE',
                GEN_TOKEN: 'GEN_TOKEN',
                COPY_TOKEN: 'COPY_TOKEN',
                GEN_COPY_TOKEN: 'GEN_COPY_TOKEN'}

# helper copied from lang.py.py_dataset.py - would NOT work with django


def get_terminal_tokens(_terminal_str):
    mode = config.data_type
    if mode == "hs":
        """
        get terminal tokens
        break words like MinionCards into [Minion, Cards]
        """

        tmp_terminal_tokens = [t for t in _terminal_str.split(' ') if len(t) > 0]
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            sub_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split(' ')
            _terminal_tokens.extend(sub_tokens)

            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]

    assert mode == "django"
    tmp_terminal_tokens = _terminal_str.split(' ')
    _terminal_tokens = []
    for token in tmp_terminal_tokens:
        if token:
            _terminal_tokens.append(token)
        _terminal_tokens.append(' ')

    return _terminal_tokens[:-1]


def alter_for_copy(ngrams, s2s_alignment_dict, grammar=None):
    # print "altering"
    # print[len(q) for q in ngrams[1:]]
    new_ngrams = []
    assert check_sanity(ngrams)
    for l in ngrams:
        new_l = []
        for ng in l:
            new_ng = []
            # print ng
            add = True
            # print ng
            for g in ng:
                if g.action_type == ACTION_NAMES[COPY_TOKEN] and g.copy_id is not None:
                    # print True
                    new_index = s2s_alignment_dict[g.copy_id][3]

                    if new_index is not None:
                        # if g.copy_id == 7:
                        #    print ng, new_index

                        g.copy_id = new_index
                        g.id = new_index
                    else:
                        add = False
                # if g.action_type == ACTION_NAMES[APPLY_RULE] and grammar is not None:
                #    print grammar.rules[g.id]
                new_ng.append(g)
                # print ng
            if add:
                new_l.append(new_ng)
        new_ngrams.append(new_l)
    assert check_sanity(new_ngrams)
    # print[len(q) for q in new_ngrams[1:]]
    return new_ngrams


def check_sanity(ngrams):
    for i in range(1, config.max_ngrams):
        for ng in ngrams[i+1]:
            found = False
            for ng2 in ngrams[i]:
                if found:
                    break
                eq = True
                for k in range(i):
                    if not ng[k].equals(ng2[k]):
                        eq = False
                    if not eq:
                        break
                if eq:
                    found = True
            if not found:
                return False
    return True


def collect_ngrams(aligned_entry, entry_index, act_sequence, unedited_words, simi_score, s2s_alignment_dict, printTree, grammar=None):
    current_ngrams = [[]]
    ngrams = [[]]
    for i in range(1, config.max_ngrams+1):
        current_ngrams.append([])
        ngrams.append([])
    current_ngram_depth = 0
    init_timestep = 0
    node = aligned_entry.parse_tree
    actions = aligned_entry.actions
    alignments = None
    print "retrieved id : %d" % aligned_entry.raw_id
    #if printTree:
    #    print aligned_entry.query
    #    print aligned_entry.parse_tree.pretty_print()
    #    for act in aligned_entry.actions:
    #        print act

    if config.use_alignment:
        alignments = aligned_entry.alignments
    # for i in range(min(len(aligned_entry.alignments), len(actions))):
    #    print aligned_entry.alignments[i], aligned_entry.query[aligned_entry.alignments[i]], actions[i]
    # print len(alignments), len(actions)
    # print aligned_entry.query
    final_timestep = aux_collect_ngrams(entry_index, actions, act_sequence, node, alignments, unedited_words, simi_score,
                                        ngrams, current_ngrams, current_ngram_depth, init_timestep, config.use_alignment)
    # print(final_timestep, len(actions))
    #assert(final_timestep == len(actions))

    return alter_for_copy(ngrams, s2s_alignment_dict, grammar)
    # return ngrams


def aux_collect_ngrams(entry_index, actions, act_sequence, node, alignments, unedited_words, simi_score, ngrams, current_ngrams, current_ngram_depth, timestep, use_alignment=False):
    # test alignment
    if timestep >= min(len(act_sequence), len(actions)):
        return timestep
    target_w = Gram(entry_index, actions[timestep], act_sequence[timestep], simi_score)
    # print actions[timestep]
    if not use_alignment or alignments[timestep] in unedited_words.values():
        current_ngram_depth = min(current_ngram_depth+1, config.max_ngrams)
        for i in range(current_ngram_depth, 0, -1):
            current_ngrams[i] = copy.deepcopy(current_ngrams[i-1])+[target_w.copy()]
            ngrams[i].append(current_ngrams[i])
    else:
        current_ngram_depth = 0
        for i in range(1, config.max_ngrams+1):
            current_ngrams[i] = []
    # print timestep
    # print current_ngram_depth
    # for i in range(current_ngram_depth, 0, -1):
    #    print i, len(current_ngrams[i])
    #    assert len(current_ngrams[i]) == i

    copy_timestep = timestep
    # print copy_timestep
    # print current_ngrams[1:3]
    assert current_ngrams[2] == [
    ] or current_ngrams[2][0].action_type == 'APPLY_RULE' or current_ngrams[2][1].action_type != 'APPLY_RULE'

    if isinstance(node, ASTNode):
        if node.children:
            for child in node.children:
                if child.children or child.value is not None:
                    # print "parent %s to child %s" % (str(node.type), str(child.type))
                    timestep = aux_collect_ngrams(entry_index, actions, act_sequence, child, alignments, unedited_words, simi_score, ngrams,
                                                  copy.deepcopy(current_ngrams), current_ngram_depth, timestep+1, use_alignment)
                    # print copy_timestep, timestep
                    # print current_ngrams[1:3]
        else:
            assert node.value is not None

            terminal_tokens = get_terminal_tokens(str(node.value))
            # print terminal_tokens
            if len(terminal_tokens) > 0:
                timestep = aux_collect_ngrams(entry_index, actions, act_sequence, terminal_tokens[1:], alignments, unedited_words, simi_score, ngrams,
                                              copy.deepcopy(current_ngrams), current_ngram_depth, timestep+1, use_alignment)
                # print copy_timestep, timestep
                # print current_ngrams[1:3]
    elif len(node) > 0:
        # print node
        timestep = aux_collect_ngrams(entry_index, actions, act_sequence, node[1:], alignments, unedited_words, simi_score, ngrams,
                                      copy.deepcopy(current_ngrams), current_ngram_depth, timestep+1, use_alignment)

    return timestep


class Gram:
    def __init__(self, entry_index, action, act_ids, score):
        self.entry_index = None
        self.action_type = None
        self.rule_id = None
        self.token_id = None
        self.copy_id = None
        self.score = None
        self.id = None
        if isinstance(action, Action):
            self.entry_index = entry_index
            self.action_type = ACTION_NAMES[action.act_type]

            if action.act_type == APPLY_RULE:
                self.rule_id = act_ids[0]
                self.id = self.rule_id

            elif action.act_type == GEN_TOKEN:
                self.token_id = act_ids[1]
                self.id = self.token_id

            elif action.act_type == COPY_TOKEN:
                self.copy_id = act_ids[2]
                self.id = self.copy_id

            else:
                assert(action.act_type == GEN_COPY_TOKEN)
                self.action_type = ACTION_NAMES[COPY_TOKEN]
                self.copy_id = act_ids[2]
                self.id = self.copy_id

            self.score = score

    def __repr__(self):

        return str((self.action_type, self.id))

    def equals(self, ng):
        return self.action_type == ng.action_type and self.rule_id == ng.rule_id and self.copy_id == ng.copy_id and self.token_id == ng.token_id and self.id == ng.id

    def copy(self):
        g = Gram(None, None, None, None)
        g.entry_index = self.entry_index
        g.action_type = self.action_type
        g.rule_id = self.rule_id
        g.token_id = self.token_id
        g.copy_id = self.copy_id
        g.id = self.id
        g.score = self.score
        return g


def insert_ngram(ng, ngram_list):
    for i, ng2 in enumerate(ngram_list):
        eq = True
        for j in range(len(ng)):
            if not ng[j].equals(ng2[j]):
                eq = False
                break
        if eq:
            if ng[0].score > ng2[0].score:
                ngram_list[i] = ng
            return
    ngram_list.append(ng)


def retrieve_translation_pieces(dataset, input_entry):
    # print input_sentence
    input_sentence = input_entry.query
    # print "input sentence"
    # print input_entry.query
    # print input_entry.parse_tree.pretty_print()
    all_ngrams = [[] for k in range(config.max_ngrams+1)]
    simi_scores = []
    for entry in dataset.examples:
        simi_scores.append(simi.simi(input_sentence, entry.query, True))
    simi_scores = np.array(simi_scores)
    simi_scores -= simi_scores.mean()
    top_indices = np.argsort(simi_scores)[-config.max_retrieved_sentences:][::-1]
    for i in top_indices:
        # print dataset.examples[i].query
        # print simi_scores[i]
        matrix, dist = simi.sentence_distance(input_sentence, dataset.examples[i].query, True)
        unedited_words, first_index_dict, second_index_dict = simi.align(
            input_sentence, dataset.examples[i].query, matrix, False, True)
        act_sequence = dataset.data_matrix["tgt_action_seq"][i]

        printTree = True
        ngrams = collect_ngrams(dataset.examples[i], i, act_sequence,
                                unedited_words, simi_scores[i], second_index_dict, printTree, grammar=dataset.grammar)
        # print "retrieved sentence"
        # print dataset.examples[i].query
        # print dataset.examples[i].parse_tree.pretty_print()
        # print simi_scores[i]

        for i in range(1, config.max_ngrams+1):
            all_ngrams[i] += ngrams[i]
    # print[len(q) for q in all_ngrams[1:]]
    max_ngrams = [[] for k in range(config.max_ngrams+1)]
    for i in range(1, config.max_ngrams+1):
        for ng in all_ngrams[i]:
            insert_ngram(ng, max_ngrams[i])
    # print[len(q) for q in max_ngrams[1:]]
    # print(input_sentence)
    return NGramSearcher(max_ngrams)


class NGramSearcher:
    def __init__(self, ngram_lists):
        self.max_ngrams = len(ngram_lists)-1
        self.total_ngrams = sum([len(q) for q in ngram_lists])
        full_list = []
        start_index = [0]
        for i in range(1, self.max_ngrams+1):
            full_list = full_list + ngram_lists[i]
            start_index.append(start_index[-1]+len(ngram_lists[i-1]))

        assert self.total_ngrams == len(full_list)
        self.ngrams_lastelt_id = []
        self.ngrams_lastelt_flag = []
        self.ngrams_score = []
        self.ngram_follows = [[] for i in range(self.total_ngrams)]
        self.indexes = dict()
        self.indexes_per_last_value = dict()

        for i, ng in enumerate(full_list):
            flag = ng[-1].action_type
            self.ngrams_lastelt_flag.append(flag)
            if flag == "APPLY_RULE":
                self.ngrams_lastelt_id.append(ng[-1].rule_id)
            elif flag == "COPY_TOKEN":
                self.ngrams_lastelt_id.append(ng[-1].copy_id)
            else:
                self.ngrams_lastelt_id.append(ng[-1].token_id)
            self.ngrams_score.append(ng[-1].score)
            l = []
            for g in ng:
                l.append(g.id)
                l.append(g.action_type)
            self.indexes[tuple(l)] = i
            if len(l) > 2:
                other_array = [self.indexes[tuple(l[:-2])]]+l[-2:]
                # print other_array
                self.indexes_per_last_value[tuple(other_array)] = i

        # self.unigrams = zip(self.ngrams_lastelt_id[:start_index[2]], self.ngrams_score[:start_index[2]], self.ngrams_lastelt_flag[:start_index[2]])

        for i in range(1, self.max_ngrams):
            for j, ng in enumerate(ngram_lists[i]):
                for k, ng2 in enumerate(ngram_lists[i+1]):
                    eq = True
                    for m in range(len(ng)):
                        if not ng[m].equals(ng2[m]):
                            eq = False
                            break
                    if eq:
                        self.ngram_follows[start_index[i]+j].append(start_index[i+1]+k)
        # for j in ngram_follows[0]:
        #    # print j, ngrams_lastelt_id[j], ngrams_lastelt_flag[j]
        # for i in range(start_index[2]):
        #     print self.ngrams_lastelt_id[i], self.ngrams_lastelt_flag[i]

    def get_keys(self, previous_keys, new_value, new_flag, verbose=False):
        new_keys = [None]

        if verbose:
            print "new unigram :"
            print new_value, new_flag
        try:
            # print(new_value, new_flag)
            index = self.indexes[(new_value, new_flag)]
            # print index
            new_keys.append(index)
            for i in range(1, self.max_ngrams):
                if verbose:
                    print(previous_keys[i], new_value, new_flag)
                index = self.indexes_per_last_value[(previous_keys[i], new_value, new_flag)]
                new_keys.append(index)
        except:
            while len(new_keys) <= self.max_ngrams:
                new_keys.append(None)
        return new_keys

    def __call__(self, keys):
        l = list()
        for k in reversed(keys[:-1]):
            if k is not None:
                for j in self.ngram_follows[k]:
                    l.append(
                        (self.ngrams_lastelt_id[j], self.ngrams_score[j], self.ngrams_lastelt_flag[j]))
            # if len(l) > 0:
            #    break
        return l


if __name__ == "__main__":
    train_data, dev_data, test_data = deserialize_from_file('../../files/aligned_hs.bin')
    for ex in test_data.examples:
        input_sentence = ex.query
        l = retrieve_translation_pieces(train_data, input_sentence)
        # print len(l[4])
        del l
