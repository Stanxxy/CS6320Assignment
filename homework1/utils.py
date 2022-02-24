import unittest
import numpy as np
from multiprocessing.pool import Pool
from nltk.tokenize import word_tokenize
# training set constructionj
# testing set construction
# IO scripts

BOS = "<s>"
EOS = "</s>"


def parse_sentence(sentence: str, pre_processeed=True) -> list:
    if pre_processeed:
        return sentence.split()
    else:
        return word_tokenize(sentence)


def parse_slice(sentence_list: list, pre_processeed=True) -> list:
    return list(map(lambda x: parse_sentence(x, pre_processeed=pre_processeed), sentence_list))


def construct_data_set(sentences_list: list, pre_processed=True, parallel_cores=4) -> list:
    temp_pool = Pool(parallel_cores)
    # put each range of sentences list into a threed to parse
    unit_size = int(np.ceil(sentences_list.__len__() / parallel_cores))
    async_tasks = []
    for i in range(parallel_cores):
        # prepare slice
        upper_bound = min((i+1) * unit_size, sentences_list.__len__())
        # print(i, unit_size, upper_bound)
        sentences_slice = sentences_list[i*unit_size: upper_bound]
        # process sentences in the slice
        async_tasks.append(temp_pool.apply_async(
            parse_slice, args=(sentences_slice, pre_processed)))

    tokenized_sentences = []
    for task in async_tasks:
        tokenized_sentences += task.get()

    return tokenized_sentences


def read_from_disk(path: str) -> list:
    res = list()
    with open(path, "r") as f:
        for line in f:
            res.append(line.strip().lower())
    return res


def compute_trigram(sentence: list) -> list:
    if sentence.__len__() == 0:
        return [BOS] * 3
    else:
        sentence = [BOS] + sentence + [EOS]
        res = []
        for i in range(sentence.__len__() - 2):
            res.append(tuple(sentence[i: i+3]))
        return res


def compute_bigram(sentence: list) -> list:
    if sentence.__len__() == 0:
        return [BOS] * 2
    else:
        sentence = [BOS] + sentence + [EOS]
        res = []
        for i in range(sentence.__len__() - 1):
            res.append(tuple(sentence[i: i+2]))
        return res


def compute_bigram_no_end_mark(sentence: list) -> list:
    sentence = sentence[:-1]  # remove "."
    if sentence.__len__() == 0:
        return [BOS] * 2
    else:
        sentence = [BOS] + sentence + [EOS]
        res = []
        for i in range(sentence.__len__() - 1):
            res.append(tuple(sentence[i: i+2]))
        return res


def compute_unigram(sentence: list) -> list:
    if sentence.__len__() == 0:
        return [BOS] * 1
    else:
        sentence = [BOS] + sentence + [EOS]
        res = []
        for i in range(sentence.__len__()):
            res.append(tuple(sentence[i: i+1]))
        return res


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.test_sentences_preprocessed = [
            "The quick fox jumps over the lazy dog .",
            "Have a great day sir ."
        ]
        self.test_sentences_un_preprocessed = [
            "Mr. Blade likes you so much.",
            "Let's go to the gym!"
        ]
        return super().setUp()

    def test_1_parse_slice(self):
        print(parse_slice(self.test_sentences_preprocessed))
        print(parse_slice(self.test_sentences_un_preprocessed, pre_processeed=False))

    def test_2_construct_data_set(self):

        sentences = [
            "This is the first sentence.",
            "This is the second.",
            "Now we copy these sentences.",
            "Just to test the multi process program",
        ] * 100
        res = construct_data_set(sentences)
        print(res[:4])

    def test_3_read_from_disk(self):
        # path = input("Please input the path of corpus.")
        path = "corpus_for_language_models.txt"
        my_list = read_from_disk(path)
        print(my_list[:2])

    def test_4_compute_trigrams(self):
        sentence = parse_sentence("The quick fox jumps over the lazy dog .")
        grams = compute_trigram(sentence)
        print("The {} gram is {}, the expected gram is{}".format(
            "first", grams[0], (BOS, "The", "quick")))
        print("The {} gram is {}, the expected gram is{}".format(
            "second", grams[1], ("The", "quick", "fox")))
        print("The {} gram is {}, the expected gram is{}".format(
            "third", grams[-1], ("dog", ".", EOS)))

    def test_5_compute_bigrams(self):
        sentence = parse_sentence("The quick fox jumps over the lazy dog .")
        grams = compute_bigram(sentence)
        print("The {} gram is {}, the expected gram is{}".format(
            "first", grams[0], (BOS, "The")))
        print("The {} gram is {}, the expected gram is{}".format(
            "second", grams[1], ("The", "quick")))
        print("The {} gram is {}, the expected gram is{}".format(
            "third", grams[-1], (".", EOS)))

    def test_6_compute_unigrams(self):
        sentence = parse_sentence("The quick fox jumps over the lazy dog .")
        grams = compute_unigram(sentence)
        print("The {} gram is {}, the expected gram is{}".format(
            "first", grams[0], (BOS,)))
        print("The {} gram is {}, the expected gram is{}".format(
            "second", grams[1], ("The",)))
        print("The {} gram is {}, the expected gram is{}".format(
            "third", grams[-1], (EOS,)))


if __name__ == "__main__":
    unittest.main()
