# build the ppmi on corpus
import utils
import unittest
import numpy as np
import pandas as pd


class POSTagger:
    def __init__(self) -> None:
        # A_matrix: k: row, v: column. dict
        ######### be careful! the prob matrix has no "." token. ########
        self.transition_prob_matrix = np.array([])
        self.transition_prob_eos = np.array([])
        self.transition_prob_bos = np.array([])
        self.observation_prob_matrix = np.array([])
        self.tag_list = []
        self.word_list = []

    def setup_trasition(self, trns_mtx: np.ndarray, obs_mtx: np.ndarray,
                        tag_bos_prob: np.ndarray, tag_eos_prob: np.ndarray, word_rows: list, tag_list: list) -> None:
        self.transition_prob_matrix = trns_mtx
        self.observation_prob_matrix = obs_mtx
        self.transition_prob_eos = tag_eos_prob
        self.transition_prob_bos = tag_bos_prob
        self.tag_list = tag_list
        self.word_list = word_rows

    def create_viterbi_table(self, bigrams: list) -> pd.DataFrame:
        # all_tags = set(self.tag_list).update(set(self.tag_list))
        viterbi = np.zeros(
            (self.tag_list.__len__(), bigrams.__len__() - 1))  # no </s>
        backpointer = np.zeros(
            (self.tag_list.__len__(), bigrams.__len__() - 1))  # no </s>
        io_col = self.word_list.index(bigrams[0][1])

        for index, prob_a in enumerate(self.transition_prob_bos):
            # when we encounter </s>, we use a specific way to compute
            prob_b = self.observation_prob_matrix[index, io_col]
            viterbi[index, 0] = prob_a * prob_b  # pi_i * b(oi)
            backpointer[index, 0] = 0

        for pos, bigram in enumerate(bigrams):
            # for the last one we process is in a specific way.
            if pos == 0 or pos == bigrams.__len__() - 1:
                continue
            if pos == bigrams.__len__() - 2:
                print()
            io_col = self.word_list.index(bigram[1])
            for col_index in range(self.tag_list.__len__()):
                candidate_probs = [0] * self.tag_list.__len__()
                prob_b = self.observation_prob_matrix[col_index, io_col]

                for v_row_index, prob_last in enumerate(viterbi[:, pos-1].reshape(-1)):
                    # if self.tag_list[v_row_index] == utils.EOS:
                    #     # backpointer[col_index, pos] = v_row_index
                    #     continue
                    # row_index = self.tag_list.index(
                    #     self.tag_list[v_row_index])
                    prob_a = self.transition_prob_matrix[v_row_index, col_index]
                    candidate_probs[v_row_index] = prob_last * prob_a * prob_b

                viterbi[col_index, pos] = np.max(candidate_probs)
                backpointer[col_index, pos] = np.argmax(candidate_probs)

        # now we finish all the pre computation
        pos = bigrams.__len__() - 1
        # last_index = self.tag_list.__len__() - 1
        candidate_probs = [0] * self.tag_list.__len__()
        for v_row_index, prob_last in enumerate(viterbi[:, pos - 1].reshape(-1)):
            # if self.tag_list[v_row_index] == utils.EOS:
            #     # backpointer[col_index, pos] = v_row_index
            #     continue
            # row_index = self.tag_list.index(
            #     self.tag_list[v_row_index])
            prob_a = self.transition_prob_eos[v_row_index]
            candidate_probs[v_row_index] = prob_last * prob_a

        output_prob = np.max(candidate_probs)
        output_ptr = np.argmax(candidate_probs)

        table = pd.DataFrame(data=viterbi, index=self.tag_list, columns=[
                             x[1] for x in bigrams][:-1])
        return table, backpointer, output_prob, output_ptr

    def list_prob_of_after_x(self, bigrams: list, step: int) -> tuple:
        # all_tags = set(self.tag_list).update(set(self.tag_list))
        viterbi = np.zeros(
            (self.tag_list.__len__(), bigrams.__len__() - 1))
        backpointer = np.zeros(
            (self.tag_list.__len__(), bigrams.__len__() - 1))
        io_col = self.word_list.index(bigrams[0][1])

        for index, prob_a in enumerate(self.transition_prob_bos):
            # when we encounter </s>, we use a specific way to compute
            # if index == self.tag_list.__len__() - 1:
            #     continue
            prob_b = self.observation_prob_matrix[index, io_col]
            viterbi[index, 0] = prob_a * prob_b  # pi_i * b(oi)
            backpointer[index, 0] = 0
        if step == 1:
            return self.tag_list + [utils.EOS], list(self.transition_prob_bos) + [0], list(
                self.observation_prob_matrix[:, io_col].reshape(-1)) + [0]  # the transition and

        transition_prob = []
        observation_prob = []
        for pos, bigram in enumerate(bigrams):
            # for the last one we process is in a specific way.
            if pos == 0 or pos == bigrams.__len__() - 1:
                continue
            io_col = self.word_list.index(bigram[1])
            if pos == step - 1:
                observation_prob = list(self.observation_prob_matrix[:, io_col].reshape(
                    -1))
            for col_index in range(self.tag_list.__len__()):
                # if self.tag_list[col_index] == utils.EOS:
                #     backpointer[col_index, pos] = col_index
                #     continue
                candidate_probs = [0] * self.tag_list.__len__()
                prob_b = self.observation_prob_matrix[col_index, io_col]

                for v_row_index, prob_last in enumerate(viterbi[:, pos-1].reshape(-1)):
                    # if self.tag_list[v_row_index] == utils.EOS:
                    #     continue
                    # row_index = self.tag_list.index(
                    #     self.tag_list[v_row_index])
                    prob_a = self.transition_prob_matrix[v_row_index, col_index]
                    candidate_probs[v_row_index] = prob_last * prob_a * prob_b

                viterbi[col_index, pos] = np.max(candidate_probs)
                backpointer[col_index, pos] = np.argmax(candidate_probs)
                if pos == step - 1:
                    prob_a_recorded = self.transition_prob_matrix[int(
                        backpointer[col_index, pos]), col_index]
                    transition_prob.append(prob_a_recorded)
            if pos == step - 1:
                # print("rerturn with transition and observation")
                # print(observation_prob.__len__())
                return self.tag_list + [utils.EOS], transition_prob + [0], observation_prob + [0]
        if step == bigrams.__len__():
            pos = step - 1
            # last_index = self.tag_list.__len__() - 1
            candidate_probs = [0] * self.tag_list.__len__()
            for v_row_index, prob_last in enumerate(viterbi[:, pos-1].reshape(-1)):
                # if self.tag_list[v_row_index] == utils.EOS:
                #     continue
                # row_index = self.tag_list.index(
                #     self.tag_list[v_row_index])
                prob_a = self.transition_prob_eos[v_row_index]
                candidate_probs[v_row_index] = prob_last * prob_a

            prob_a_recorded = np.max(candidate_probs)
            back_ptr = np.argmax(candidate_probs)

            # prob_a_recorded = self.transition_prob_matrix[int(
            #     backpointer[col_index, pos]), col_index]
            transition_prob.append(prob_a_recorded)

            return self.tag_list + [utils.EOS], [0] * self.tag_list.__len__() + transition_prob, \
                [0] * bigrams.__len__() + observation_prob + [1]

    def find_prob(self, bigrams: list) -> tuple:
        table, bck_pointer, output_prob, output_ptr = self.create_viterbi_table(
            bigrams=bigrams)

        def find_path(tags: list, backpointer: np.ndarray, output_ptr: int) -> list:
            path = []
            n = backpointer[0].__len__()
            # int(backpointer[n-1, tags.__len__() - 1])
            last_pointer = output_ptr
            path.insert(0, tags[last_pointer])
            for i in range(n-1, 0, -1):
                new_pointer = int(backpointer[last_pointer, i])
                # print("This is new pointer:", new_pointer)
                path.insert(0, tags[new_pointer])
                last_pointer = new_pointer
            return path

        path = find_path(table.index, bck_pointer, output_ptr=output_ptr)
        return output_prob, path


class TestPOSTagger(unittest.TestCase):
    def setUp(self) -> None:
        matrix_A = np.array([
            [0, 0.58, 0, 0, 0, 0.42, 0, 0],  # , 0],
            [0, 0.07, 0, 0.05, 0.32, 0, 0, 0.25],  # 0.11],
            [0.07, 0.08, 0, 0, 0, 0, 0.2, 0.61],  # 0.13],
            [0.2, 0.3, 0, 0, 0, 0.24, 0.15, 0.11],  # 0],
            [0.18, 0.22, 0, 0, 0.2, 0.07, 0.16, 0.11],  # 0.06],
            [0, 0.88, 0, 0, 0, 0.12, 0, 0],  # 0],
            [0, 0, 0, 0.22, 0.28, 0.39, 0.1, 0],  # 0.01],
            [0.57, 0.28, 0, 0, 0, 0.15, 0, 0],  # 0]
        ])
        # tag_list = ["DT", "NN", "VB",
        #                 "VBZ", "VBN", "JJ", "RB", "IN"]
        # .tag_list = ["DT", "NN", "VB", "VBZ",
        #                 "VBN", "JJ", "RB", "IN"]
        tag_list = ["DT", "NN", "VB", "VBZ",
                    "VBN", "JJ", "RB", "IN"]
        tag_bos_prob = np.array([0.38, 0.32, 0.04, 0, 0, 0.11, 0.01, 0.14])
        tag_eos_prob = np.array([0, 0.11, 0.13, 0, 0.06, 0, 0.01, 0])
        matrix_B = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.69, 1, 0.88, 1, 0, 0, 0.01, 0.66, 0.38, 0, 0, 0],
            [0, 0, 0.31, 1, 0.12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0.99, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.34, 0.62, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ])

        word_list = ["a", "the", "chair", "chairman", "board", "road", "is",
                     "was", "found", "middle", "bold", "completely", "in", "of"]

        s1 = "The chairman of the board is completely bold."
        s2 = "A chair was found in the middle of the road."
        self.bigram_1 = utils.compute_bigram_no_end_mark(
            utils.parse_sentence(s1.lower(), pre_processeed=False))
        self.bigram_2 = utils.compute_bigram_no_end_mark(
            utils.parse_sentence(s2.lower(), pre_processeed=False))
        # print(self.bigram_1)
        # self.positive_sentence = "Richard W. Lock , retired vice president and treasurer of Owens-Illinois Inc. , was named a director of this transportation industry supplier , increasing its board to six members ."
        # self.negative_sentence = "Today is a sunny day!"
        self.model = POSTagger()
        self.model.setup_trasition(
            matrix_A, matrix_B, tag_bos_prob, tag_eos_prob, word_list, tag_list)
        return super().setUp()

    def test_1_list_prob_after_x(self):
        _, prob_a, prob_b = self.model.list_prob_of_after_x(self.bigram_1, 1)
        print("tag, status prob, observation prob")
        for i in range(self.model.tag_list.__len__()):
            print(self.model.tag_list[i], prob_a[i], prob_b[i])

        _, prob_a, prob_b = self.model.list_prob_of_after_x(self.bigram_1, 3)
        # print(prob_b)
        print("tag, status prob, observation prob")
        for i in range(self.model.tag_list.__len__()):
            # print(i)
            print(self.model.tag_list[i], prob_a[i], prob_b[i])

        _, prob_a, prob_b = self.model.list_prob_of_after_x(self.bigram_1, 8)
        # print(prob_a.__len__())
        print("tag, status prob, observation prob")
        for i in range(self.model.tag_list.__len__()):
            print(self.model.tag_list[i], prob_a[i], prob_b[i])

    def test_2_test_viterbi_table(self):
        viterbi_table, pointer_table, output_prob, bck_ptr = self.model.create_viterbi_table(
            self.bigram_1)
        print(viterbi_table)
        print(pointer_table)

    def test_3_prob_and_path(self):
        likelihood, tag_list = self.model.find_prob(self.bigram_1)
        print("The likelyhood is {}".format(likelihood))
        print("tag is : {}".format(tag_list))


if __name__ == "__main__":
    unittest.main()
