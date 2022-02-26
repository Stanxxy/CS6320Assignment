# Document for CS6320 Assignment 1

- This is the document for the code for Assignment 1 of CS6320 22 Spring at UTD

## Design

- frontend.py

  - Methods:
    - read_file - read files from disk.
    - construct_models - constract three types of models: basic trigram model, trigram model with Laplace smoothing and trigram model with Katz smoothing
    - read_problem1_cases - read the test sentence for problem 1 from user input
    - print_ans_1_a_1 - print the answer of the first sub problemm of problem 1
    - print_ans_1_a_234 - print the answer of the second, third and fourth sub problems of problem 1
    - print_ans_1_a_5 - print the answer of the fifth sub problems of problem 1
    - read_problem2_cases - read the test cases for problem 2 from user input
    - construct_model - construct the model for the first sub problem of problem 2
    - print_ans_2_1 - print the answer of the first sub problems of problem 2
    - read_problem2_constraints_and_cases - read the constraints words and cases from the second sub problem of problem 2
    * retrain_model - retrain the model for the second sub problem of problem 2
    * print_ans_2_2 - print the answer of the second sub problems of problem 2
    * prepare_case_for_ans_3 - define the inputs and test cases for problem3
    * print_ans_3 - print the answer for problem 3

- utils.py

  - Methods:
    - parse_sentence - parse the sentence. The sentence must have been tokenized (puctuations has been separated with )
    - parse_slice - parse the list of sentence. The list must contains sentences that has been tokenized.
    - construct_data_set - construct a sentence dataset with multiprocess parsing
    - read_from_disk - read sentence from disk. The format is a corpus format with each sentence in a line
    - compute_trigram - compute the trigrams for a list of tokens. Return a list of trigrams which are represented by tuple
    - compute_bigram - computer the bigrams for a list of tokens. Return similary to compute_trigrams
    - compute_bigram_no_end_mark - compute the bigrams without the last token (usually the token is punctuation). This is used for HMM model where the punctuation appears neither in the transition matrix nor in the observation matrix.
    - compute_unigram - compute the unigrams for a list of tokens. Return similary to compute_trigrams.
  - Classes
    - TestUtils - the class for unit test
  - Constants
    - BOS - the token for begin of sentence
    - EOS - the token for end of sentence

* language_models.py

  - Classes
    - GramModel - the basic language model which use trigram counts
    - LaplaceModel - the enhanced language model with laplace smoothing
    - KatzModel - the enhanced language model with Katz smoothing
    - TestModels - the class for unit test

* semantic_models.py

  - Classes
    - PPMI - the semantic model class
      - generate_int_default_int/generate_int_default_float - return the default dictionary whose default valud is a default dictionary with int/float as the default value
      - train - train the model with given corpus
      * train_with_context_constraint - train the model with context that must contain certain words
      * predict - predict the semantic distance between word pairs
      * predict_with_constraint_context - predict the semantic distance between word pairs with a constraint model
    - TestPPMI - the class for unit test

* POS_tagging_model
  - Classes
    - POSTagger - the POS tagger
