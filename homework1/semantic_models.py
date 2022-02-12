# build the ppmi on corpus
from collections import defaultdict
import utils
import unittest
import numpy as np


class PPMI:
    def __init__(self, training_set: list) -> None:
        self.training_set = training_set

    def generate_int_default_int(self):
        return defaultdict(int)

    def generate_int_default_float(self):
        return defaultdict(float)

    def train(self) -> None:
        # code for building PPMI model
        counter_table_r = defaultdict(self.generate_int_default_int)
        counter_table_c = defaultdict(self.generate_int_default_int)
        total = 0
        for tokens in self.training_set:
            padded_tokens = ["NIL"] * 5 + tokens + ["NIL"] * 5
            token_len = tokens.__len__()
            for i in range(token_len):
                # start_index and end_index are inclusive
                start_index = i
                center_index = start_index+5
                end_index = center_index+5+1
                left_slice = padded_tokens[start_index: center_index]
                right_slice = padded_tokens[center_index+1: end_index]
                for i in range(left_slice.__len__()):
                    counter_table_r[padded_tokens[center_index]
                                    ][left_slice[i]] += 1
                    counter_table_r[padded_tokens[center_index]
                                    ][right_slice[i]] += 1
                    counter_table_c[left_slice[i]
                                    ][padded_tokens[center_index]] += 1
                    counter_table_c[right_slice[i]
                                    ][padded_tokens[center_index]] += 1

        self.counter_table_r = counter_table_r
        self.counter_table_c = counter_table_c
        for v in self.counter_table_r.values():
            for v2 in v.values():
                total += v2
        self.total = total

    def train_with_context_constraint(self, constraint_words: list) -> None:
        # code for building PPMI model
        counter_table_r = defaultdict(self.generate_int_default_int)
        counter_table_c = defaultdict(self.generate_int_default_int)
        total = 0
        for tokens in self.training_set:
            padded_tokens = ["NIL"] * 5 + tokens + ["NIL"] * 5
            token_len = tokens.__len__()
            for i in range(token_len):
                # start_index and end_index are inclusive
                start_index = i
                center_index = start_index+5
                end_index = center_index+5+1
                left_slice = padded_tokens[start_index: center_index]
                right_slice = padded_tokens[center_index+1: end_index]

                flag = True
                for w in constraint_words:
                    if flag:
                        if not (w in left_slice or w in right_slice):
                            flag &= False
                    else:
                        break

                if flag:
                    # print("slice on left:", left_slice)
                    # print("slice on right:", right_slice)
                    for i in range(left_slice.__len__()):
                        counter_table_r[padded_tokens[center_index]
                                        ][left_slice[i]] += 1
                        counter_table_r[padded_tokens[center_index]
                                        ][right_slice[i]] += 1
                        counter_table_c[left_slice[i]
                                        ][padded_tokens[center_index]] += 1
                        counter_table_c[right_slice[i]
                                        ][padded_tokens[center_index]] += 1

        self.counter_table_constraint_r = counter_table_r
        self.counter_table_constraint_c = counter_table_c

        for v in self.counter_table_constraint_r.values():
            for v2 in v.values():
                total += v2
        self.total = total

    def predict(self, w: str, c: str) -> float:
        # w is the word and c is the contest
        # code for predict the semantic similarity between two words
        fij = self.counter_table_r[w][c] / self.total
        fi_star = np.sum(list(self.counter_table_r[w].values())) / self.total
        fj_star = np.sum(list(self.counter_table_c[c].values())) / self.total
        # print("fij: {}, fi*: {}, f*j: {}".format(fij, fi_star, fj_star))
        PPMI_value = 0 if fij == 0 else max(
            np.log2(fij / (fi_star * fj_star)), 0)
        return PPMI_value

    def predict_with_constraint_context(self, w: str, c: str) -> float:
        # w is the word and c is the contest
        # code for predict the semantic similarity between two words
        fij = self.counter_table_constraint_r[w][c] / self.total
        fi_star = np.sum(
            list(self.counter_table_constraint_r[w].values())) / self.total
        fj_star = np.sum(
            list(self.counter_table_constraint_c[c].values())) / self.total
        # print("fij: {}, fi*: {}, f*j: {}".format(fij, fi_star, fj_star))
        PPMI_value = 0 if fij == 0 else max(
            np.log2(fij / (fi_star * fj_star)), 0)
        return PPMI_value


class TestPPMI(unittest.TestCase):
    def setUp(self) -> None:
        path = "corpus_for_language_models.txt"
        sentences = utils.read_from_disk(path)
        self.corpus = utils.construct_data_set(sentences_list=sentences)
        # self.positive_sentence = "Richard W. Lock , retired vice president and treasurer of Owens-Illinois Inc. , was named a director of this transportation industry supplier , increasing its board to six members ."
        # self.negative_sentence = "Today is a sunny day!"
        self.model = PPMI(self.corpus)
        return super().setUp()

    def test_1_PPMI_normal_train(self):
        self.model.train()
        predict_cases = [
            ("chairman", "said"),
            ("chairman", "of"),
            ("company", "board"),
            ("company", "said"),
        ]
        prob = []
        for w, c in predict_cases:
            prob.append(self.model.predict(w, c))
            print(prob[-1])

    def test_2_PPMI_train_with_context_constraint(self):
        constraint_words = ["said", "of", "board"]
        self.model.train_with_context_constraint(
            constraint_words=constraint_words)
        predict_cases = [
            ("chairman", "company"),
            ("company", "sales"),
            ("company", "economy"),
        ]
        prob = []
        for w, c in predict_cases:
            prob.append(self.model.predict_with_constraint_context(w, c))
            print(prob[-1])


if __name__ == "__main__":
    unittest.main()
