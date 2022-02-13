import utils
import language_models
import semantic_models
import os
# user interface
# problem seelction
# problem validation

# Do IO and Pre processing


def read_file() -> tuple:
    file_name = input("please input the name of the corpus file.\n")
    if not os.path.exists(file_name):
        print("wrong path. check your path again.")
        exit()

    seg_flag = input("Has your corpus been tokenized?[y/n]")
    if seg_flag.lower() == 'y':
        seg_flag = True
    elif seg_flag.lower() == 'n':
        seg_flag = False
    else:
        print("please input number that makes sense.")
        exit()

    return utils.read_from_disk(file_name), seg_flag

# Problem1


def construct_models(training_set) -> tuple:
    trigram_model = language_models.GramModel(training_set=training_set)
    trigram_model.train()
    laplace_model = language_models.LaplaceModel(training_set=training_set)
    laplace_model.train()
    katz_backoff_model = language_models.KatzModel(training_set=training_set)
    katz_backoff_model.train()
    return trigram_model, laplace_model, katz_backoff_model

# A


def read_problem1_cases() -> list:
    problem_cases = []
    s = input("Please input test sentences. put q to end input.\n")
    while s != "q":
        problem_cases.append(s)
        s = input()
    return problem_cases


def print_ans_1_a_1(s_list) -> list:
    trigrams_list = []
    # print trigram
    for s in s_list:
        s_i_list = utils.parse_sentence(s)
        trigram_i = utils.compute_trigram(s_i_list)
        trigrams_list.append(trigram_i)

    print("Trigram list is: ")
    for index, trigrams in enumerate(trigrams_list):
        print("sentence {} has trigrams:".format(index))
        for trigram in trigrams:
            print(trigram, end="\t")
        print()
    return trigrams_list


def print_ans_1_a_234(trigram_model: language_models.GramModel,
                      laplace_model: language_models.LaplaceModel,
                      katz_backoff_model: language_models.KatzModel) -> None:

    # # obtain counts and probs for baseline model
    # count_df = trigram_model.get_count_table()
    # counts_file_name = "trigram_counts_without_smoothing.csv"
    # count_df.to_csv(counts_file_name)
    # print("trigram counts matrix has been saved to {}".format(counts_file_name))
    # prob_df = trigram_model.get_prob_table()
    # prob_file_name = "trigram_probs_without_smoothing.csv"
    # prob_df.to_csv(prob_file_name)
    # print("trigram probs matrix has been saved to {}".format(prob_file_name))

    # obtain counts and probs for laplace model
    count_df = laplace_model.get_count_table()
    counts_file_name = "laplace_smoothed_counts.csv"
    count_df.to_csv(counts_file_name)
    print("laplace-smoothed count matrix has been saved to {}".format(counts_file_name))
    prob_df = laplace_model.get_prob_table()
    prob_file_name = "laplace_probs_without_smoothing.csv"
    prob_df.to_csv(prob_file_name)
    print("laplace probs matrix has been saved to {}".format(prob_file_name))

    reconstitiuted_count_df = laplace_model.get_reconstituted_count_table()
    re_counts_file_name = "laplace_reconstituted_counts.csv"
    reconstitiuted_count_df.to_csv(re_counts_file_name)
    print("laplace reconstituted matrix has been saved to {}".format(
        re_counts_file_name))

    # print how many times did I compute bigram and unigram
    # prob_df, num_bigrams, num_unigrams = katz_backoff_model.get_prob_table_and_compute_number()
    # prob_file_name = "katz_probs_without_smoothing.csv"
    # prob_df.to_csv(prob_file_name)
    # print("katz probs matrix has been saved to {}".format(prob_file_name))
    # print("We compute {} bigrams probabilities and {} unigrams probabilities".format(
    #     num_bigrams, num_unigrams))


def print_ans_1_a_5(trigram_model:  language_models.GramModel,
                    laplace_model: language_models.LaplaceModel,
                    katz_backoff_model: language_models.KatzModel, s_list) -> None:
    prob_base_list = []
    prob_laplace_list = []
    prob_katz_list = []
    tokenized_list = []
    for s in s_list:
        s_i_list = utils.parse_sentence(s, pre_processeed=False)
        tokenized_list.append(s_i_list)
        p_base_s_i = trigram_model.predict(s_i_list)
        p_laplace_s_i = laplace_model.predict(s_i_list)
        p_katz_backoff_s_i = katz_backoff_model.predict(s_i_list)
        prob_base_list.append(p_base_s_i)
        prob_laplace_list.append(p_laplace_s_i)
        prob_katz_list.append(p_katz_backoff_s_i)
    for i, s in enumerate(s_list):
        print("The probability of sentence {} predicted by for trigram model is: {}".format(
            i, prob_base_list[i]))
    for i, s in enumerate(s_list):
        print("The probability of sentence {} predicted by for trigram model is: {}".format(
            i, prob_laplace_list[i]))
    for i, s in enumerate(s_list):
        print("The probability of sentence {} predicted by for trigram model is: {}".format(
            i, prob_katz_list[i]))


# Problem2
def read_problem2_cases() -> list:
    problem_cases = []
    s = input(
        "Please input the word pairs. separate with \",\". Input q to end input.\n")
    while s != "q":
        problem_cases.append(tuple([word.strip() for word in s.split(",")]))
        s = input()
    return problem_cases


# 1
def construct_model(tokenized_corpus: list) -> semantic_models.PPMI:
    model = semantic_models.PPMI(tokenized_corpus)
    model.train()
    return model


def print_ans_2_1(PPMI_model: semantic_models.PPMI, word_pair_list: list) -> None:
    similarities = []
    for pair in word_pair_list:
        similarity = PPMI_model.predict(pair[0], pair[1])
        similarities.append(similarity)
        print("for word {} and {}, their similarity valud is {}".format(
            pair[0], pair[1], similarities))

# 2


def read_problem2_constraints_and_cases() -> tuple:
    constraint_words = []
    s = input(
        "Please input the constraint words. Input q to end input.\n")
    while s != "q":
        constraint_words.append(s.strip())
        s = input()

    test_cases = []
    s = input(
        "Please input the constraint words. separate with \",\". Input q to end input.\n")
    while s != "q":
        test_cases.append(tuple([word.strip() for word in s.split(",")]))
        s = input()
    return constraint_words, test_cases


def retrain_model(model: semantic_models.PPMI, constraint_words: list) -> semantic_models.PPMI:
    model.train_with_context_constraint(constraint_words)
    return model


def print_ans_2_2(PPMI_model: semantic_models.PPMI, word_pair_list: list) -> None:
    similarities = []
    for pair in word_pair_list:
        similarity = PPMI_model.predict_with_constraint_context(
            pair[0], pair[1])
        similarities.append(similarity)
        print("Given the context constraint, for word {} and {}, their similarity valud is {}".format(
            pair[0], pair[1], similarities))


if __name__ == "__main__":
    # Corpus preparation

    corpus, preprocessed = read_file()
    tokenized_corpus = utils.construct_data_set(
        corpus, pre_processed=preprocessed)
    # problem1 A
    model1, model2, model3 = construct_models(tokenized_corpus)
    problem1_cases = read_problem1_cases()
    print_ans_1_a_1(problem1_cases)
    print_ans_1_a_234(model1, model2, model3)
    print_ans_1_a_5(model1, model2, model3, problem1_cases)
    # problem1 B
    print("For problem 1B, please see the ipynb.")
    # problem2
    semantic_model = construct_model(tokenized_corpus)
    problem2_cases = read_problem2_cases()
    print_ans_2_1(semantic_model, problem2_cases)
    constraint_words, problem2_new_cases = read_problem2_constraints_and_cases()
    retrain_model(semantic_model, constraint_words)
    print_ans_2_2(semantic_model, problem2_new_cases)
    # problem3
