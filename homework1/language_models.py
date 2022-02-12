import unittest
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils
from collections import defaultdict
from random import shuffle
# x gram model
# x gram + Laplacian smoothing
# x gram + katz backoff


class baseModel:
    def __init__(self, training_set: list) -> None:
        self.training_set = training_set

    def generate_int_default_int(self):
        return defaultdict(int)

    def generate_int_default_float(self):
        return defaultdict(float)

    def train(self) -> None:
        pass

    def predict(self, sentence: list) -> float:
        pass

    def get_count_table(self) -> pd.DataFrame:
        pass

    def get_prob_table(self) -> pd.DataFrame:
        pass


class GramModel(baseModel):
    def __init__(self, training_set: list, gram=3) -> None:
        self.training_set = training_set
        self.gram = gram

    def train(self) -> None:
        context_to_wn = defaultdict(self.generate_int_default_int)
        wordn = set()
        prob_wn_conditional_on_context = defaultdict(
            self.generate_int_default_float)

        for sentence in self.training_set:
            # we have a list of tokens
            trigrams = utils.compute_trigram(sentence)
            for trigram in trigrams:
                wn = trigram[2]
                context = trigram[:2]
                wordn.add(wn)
                context_to_wn[context][wn] += 1
        self.count = context_to_wn
        self.wordn = wordn

        for context, words in context_to_wn.items():
            total = np.sum(list(words.values()))
            for word in words.keys():
                prob_wn_conditional_on_context[context][word] = words[word] / total
        self.model_prob = prob_wn_conditional_on_context

    def predict(self, sentence: list) -> float:
        trigrams = utils.compute_trigram(sentence)
        cumulative_prob = 1
        for trigram in trigrams:
            wn = trigram[2]
            context = trigram[:2]
            cumulative_prob *= self.model_prob[context][wn]
        return cumulative_prob

    def log_predict(self, sentence: list) -> float:
        raw_prob = self.predict(sentence)
        return 0 if raw_prob == 0 else np.log(raw_prob)

    def get_count_table(self) -> pd.DataFrame:
        rows = []
        indices = []
        for context in self.count.keys():
            indices.append(context)
            row = []
            for word in self.wordn:
                val = self.count[context][word]
                row.append(val)
            rows.append(row)
        # convert to dataframe
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.wordn)
        return ans

    def get_prob_table(self) -> pd.DataFrame:
        rows = []
        indices = []
        for context in self.model_prob.keys():
            indices.append(context)
            row = []
            for word in self.wordn:
                val = self.count[context][word]
                row.append(val)
            rows.append(row)
        # convert to dataframe
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.wordn)
        return ans


class LaplaceModel(baseModel):
    def __init__(self, training_set: list, gram=3) -> None:
        self.training_set = training_set
        self.gram = gram

    def train(self) -> None:
        context_to_wn = defaultdict(self.generate_int_default_int)
        vocabulary = set()
        wordn = set()

        # compute counts
        for sentence in self.training_set:
            # we have a list of tokens
            trigrams = utils.compute_trigram(sentence)
            vocabulary.update(sentence)
            for trigram in trigrams:
                wn = trigram[2]
                context = trigram[:2]
                wordn.add(wn)
                context_to_wn[context][wn] += 1
        self.count = context_to_wn
        self.wordn = wordn
        self.V = vocabulary.__len__()

    def predict(self, sentence: list) -> float:
        trigrams = utils.compute_trigram(sentence)
        cumulative_prob = 1
        for trigram in trigrams:
            wn = trigram[2]
            context = trigram[:2]
            base_val = self.model_prob[context][wn]
            # if base_val == float():
            #     total = np.sum(list(self.count[context].values()))
            #     base_val = 1 / (total + self.V)
            cumulative_prob *= base_val
        return cumulative_prob

    def log_predict(self, sentence: list) -> float:
        raw_prob = self.predict(sentence)
        return 0 if raw_prob == 0 else np.log(raw_prob)

    def get_count_table(self) -> pd.DataFrame:
        rows = []
        indices = []
        print("start to compute count table")
        for context in tqdm(self.count.keys()):
            indices.append(context)
            row = []
            for word in self.wordn:
                # self.count[context][word] += 1
                val = self.count[context][word] + 1
                row.append(val)
            rows.append(row)
        print("Finish compute count table")
        # convert to dataframe
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.wordn)
        return ans

    def get_reconstituted_count_table(self) -> pd.DataFrame:
        rows = []
        indices = []
        print("start to compute rec table")
        for context in tqdm(self.count.keys()):
            indices.append(context)
            row = []
            total = np.sum(list(self.count[context].values()))
            for word in self.wordn:

                val = total * self.model_prob[context][word]
                # val = self.count[context][word]
                # if val == 0:
                #     val = 1

                #     val = self * total / (total + self.V)

                row.append(val)
            rows.append(row)
        print("Finish compute rec table")
        # convert to dataframe
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.wordn)

        return ans

    def get_prob_table(self) -> pd.DataFrame:

        # print("start to compute prob table")
        # for context, words in tqdm(context_to_wn.items()):

        #     for word in words.keys():
        #         prob_wn_conditional_on_context[context][word] = \
        #             (words[word] + 1) / (total + self.V)
        prob_wn_conditional_on_context = defaultdict(
            self.generate_int_default_float)
        rows = []
        indices = []
        n = self.wordn.__len__()
        print("start to compute prob table")
        for context, words in tqdm(self.count.items()):
            total = np.sum(list(words.values()))
            indices.append(context)
            row = [None] * n
            for i, word in enumerate(self.wordn):
                val = (self.count[context][word] + 1) / (total + n)
                prob_wn_conditional_on_context[context][word] = val
                # val = self.count[context][word]
                # row.append(val)
                # if val == float():
                #     total = np.sum(list(self.count[context].values()))
                #     val = (val + 1) / (total + self.V)
                row[i] = val
            rows.append(row)
        self.model_prob = prob_wn_conditional_on_context
        # convert to dataframe
        print("Finish compute prob table")
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.wordn)
        return ans


class KatzModel(baseModel):
    def __init__(self, training_set: list, gram=3) -> None:
        self.training_set = training_set
        self.gram = gram

    def train(self) -> None:
        trigram_context_to_wn = defaultdict(self.generate_int_default_int)
        bigram_context_to_wn = defaultdict(self.generate_int_default_int)
        unigram_context_to_wn = defaultdict(int)
        vocabulary = set()
        wordn_3 = set()
        wordn_2 = set()
        trigram_wn_conditional_on_context = defaultdict(
            self.generate_int_default_float)
        bigram_wn_conditional_on_context = defaultdict(
            self.generate_int_default_float)
        unigram_wn_conditional_on_context = defaultdict(float)
        print("start to compute raw count for all grams")
        for sentence in tqdm(self.training_set):
            # we have a list of tokens
            trigrams = utils.compute_trigram(sentence)
            bigrams = utils.compute_bigram(sentence)
            unigrams = utils.compute_unigram(sentence)
            vocabulary.update(sentence)
            for trigram in trigrams:
                wn = trigram[2]
                context = trigram[:2]
                wordn_3.add(wn)
                trigram_context_to_wn[context][wn] += 1
            for bigram in bigrams:
                wn = bigram[1]
                context = bigram[:1]
                wordn_2.add(wn)
                bigram_context_to_wn[context][wn] += 1
            for unigram in unigrams:
                unigram_context_to_wn[unigram[0]] += 1
        self.count_trigram = trigram_context_to_wn
        self.count_bigram = bigram_context_to_wn
        self.count_unigram = unigram_context_to_wn
        self.word_3 = wordn_3
        self.wordn_2 = wordn_2.__len__()
        self.wordn_3 = wordn_3.__len__()
        self.count_xgram = [self.count_unigram,
                            self.count_bigram, self.count_trigram]

        self.V = vocabulary.__len__()
        print("finish computing raw count for all grams")

        print("start to compute prob table for all grams")
        for context, words in trigram_context_to_wn.items():
            total = np.sum(list(words.values()))
            for word in words.keys():
                trigram_wn_conditional_on_context[context][word] = (
                    words[word] + 1) / (total + self.wordn_3)

        for context, words in bigram_context_to_wn.items():
            total = np.sum(list(words.values()))
            for word in words.keys():
                bigram_wn_conditional_on_context[context][word] = (
                    words[word] + 1) / (total + self.wordn_2)

        for word, count in unigram_context_to_wn.items():
            unigram_wn_conditional_on_context[word] = (count + 1) / self.V

        self.model_trigram_prob = trigram_wn_conditional_on_context
        self.model_bigram_prob = bigram_wn_conditional_on_context
        self.model_unigram_prob = unigram_wn_conditional_on_context

        self.model_xgram_prob = [self.model_unigram_prob,
                                 self.model_bigram_prob,
                                 self.model_trigram_prob]
        print("finish computing prob table for all grams")

    def get_prob_table_and_compute_number(self) -> tuple:
        rows = []
        indices = []
        bigram_prob_counter = 0
        unigram_prob_counter = 0
        print("start to compute total prob table for all grams")
        for context in tqdm(self.count_trigram.keys()):
            indices.append(context)
            row = []
            for word in self.word_3:
                virtual_tuple = tuple(list(context) + [word])
                val = self.back_off_prob_compute(virtual_tuple)
                bigram_prob_counter += self.count_bigrams_compute(
                    virtual_tuple)
                unigram_prob_counter += self.count_unigrams_compute(
                    virtual_tuple)
                row.append(val)
            rows.append(row)
        # convert to dataframe
        print("finish to compute total count for all grams")
        ans = pd.DataFrame(data=np.array(
            rows), index=indices, columns=self.word_3)
        return ans, bigram_prob_counter, unigram_prob_counter

    def back_off_prob_compute(self, gram: tuple, index=3) -> float:
        context = gram[:-1]
        word = gram[-1]
        # print(context)
        # print(word)
        if index == 1 or self.count_xgram[index - 1][context][word] != 0:
            if index == 1 and self.count_xgram[index - 1][context] == 0:
                val = 1 / self.V
            else:
                if index == 1:
                    val = self.model_xgram_prob[index - 1][context]
                else:
                    val = self.model_xgram_prob[index - 1][context][word]
            return val
        else:
            beta = self.compute_beta(context, index=index)
            # n-1 gram
            alpha = self.compute_alpha(beta, gram[1:-1], index=index)
            return alpha * self.back_off_prob_compute(gram[1:], index=index-1)

    def compute_beta(self, context: tuple, index=3) -> float:
        # compute the alpha
        return 1 - np.sum(list(self.model_xgram_prob[index-1][context].values()))

    def compute_alpha(self, beta: float, context: tuple, index=3) -> float:
        if index == 2:
            return beta / (1 - self.model_xgram_prob[index-1-1][context])
        else:
            return beta / (1 - np.sum(list(self.model_xgram_prob[index-1-1][context].values())))

    def predict(self, sentence: list) -> float:
        trigrams = utils.compute_trigram(sentence)
        cumulative_prob = 1
        for trigram in trigrams:
            cumulative_prob *= self.back_off_prob_compute(trigram)
        return cumulative_prob

    def log_predict(self, sentence: list) -> float:
        return 0 if self.predict(sentence) == 0 else np.log(self.predict(sentence))

    def count_bigrams_compute(self, trigram) -> int:
        def recursive_bigrams_count(gram: tuple) -> None:
            context = gram[:-1]
            word = gram[-1]
            if self.count_xgram[2][context][word] != 0:
                return 0
            elif self.count_xgram[1][context[1:]][word] != 0:
                return 1
            else:
                return 0

        res = recursive_bigrams_count(trigram)
        return res

    def count_unigrams_compute(self, trigram) -> int:
        def recursive_unigrams_count(gram: tuple) -> None:
            context = gram[:-1]
            word = gram[-1]
            if self.count_xgram[2][context][word] != 0 and self.count_xgram[1][context[1:]][word] != 0:
                return 0
            else:
                return 1
        res = recursive_unigrams_count(trigram)
        return res


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        path = "corpus_for_language_models.txt"
        sentences = utils.read_from_disk(path)
        self.corpus = utils.construct_data_set(sentences_list=sentences)
        self.positive_sentence = "Richard W. Lock , retired vice president and treasurer of Owens-Illinois Inc. , was named a director of this transportation industry supplier , increasing its board to six members ."
        positive_tokens = utils.parse_sentence(self.positive_sentence)
        shuffle(positive_tokens)
        self.negative_sentence = " ".join(positive_tokens)
        return super().setUp()

    # def test_1_GramModel(self):
    #     model = GramModel(self.corpus)
    #     model.train()
    #     count_tb = model.get_count_table()
    #     prob_tb = model.get_prob_table()
    #     predicted_positive = model.predict(
    #         utils.parse_sentence(self.positive_sentence))
    #     predicted_negative = model.predict(
    #         utils.parse_sentence(self.negative_sentence))
    #     print("The head of counter table:\n", count_tb.head())
    #     print("The head of probability table:\n", prob_tb.head())
    #     print("positive sentence likelihood {}".format(predicted_positive))
    #     print("negative sentence likelihood {}".format(predicted_negative))

    # def test_2_LaplaceModel(self):
    #     model = LaplaceModel(self.corpus)
    #     model.train()
    #     count_tb = model.get_count_table()
    #     prob_tb = model.get_prob_table()
    #     reconstituted_tb = model.get_reconstituted_count_table()
    #     predicted_positive = model.predict(
    #         utils.parse_sentence(self.positive_sentence))
    #     predicted_negative = model.predict(
    #         utils.parse_sentence(self.negative_sentence))
    #     print("The head of counter table:\n", count_tb.head())
    #     print("The head of probability table:\n", prob_tb.head())
    #     print("The head of reconstituted table:\n", reconstituted_tb.head())
    #     print("positive sentence likelihood {}".format(predicted_positive))
    #     print("negative sentence likelihood {}".format(predicted_negative))

    def test_3_KatzModel(self):
        model = KatzModel(self.corpus)
        model.train()
        # count_tb = model.get_count_table()
        # prob_tb, bigram_prob_counter, unigram_prob_counter = model.get_prob_table_and_compute_number()
        predicted_positive = model.predict(
            utils.parse_sentence(self.positive_sentence))
        predicted_negative = model.predict(
            utils.parse_sentence(self.negative_sentence))
        # print("The head of counter table:\n", count_tb.head())
        # print("The head of probability table:\n", prob_tb.head())
        print("positive sentence likelihood {}".format(predicted_positive))
        print("negative sentence likelihood {}".format(predicted_negative))
        # print("{} bigrams are computed. ".format(bigram_prob_counter))
        # print("{} unigrams are computed. ".format(unigram_prob_counter))


if __name__ == "__main__":
    unittest.main()
