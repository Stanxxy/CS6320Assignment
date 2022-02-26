# CS6320 Assignment 1

- Name: Sennan Liu
- Email: Sennan.Liu@UTDallas.edu
- This is the assignment repo for first assignment for NLP course 22 spring

## Brief intro for this assignment

- 3 problems has been solved in this homework
- supportive packages including:
  - pandas
  - numpy

* description for each script:

  - frontend.py is for the main process that is called to solve all problems in a one way flow
  - language_model.py is for problem 1. With this script you can build language models can be build by this script.
  - semantic_models.py is for problem 2. With this cript you can build the semantic building the PPMI similarity and compute the similarity of any word pairs you want,
  - POS_tagging_model.py is for problem 3. Code including pos tagging dynamic programming, tag prediction.

  * utils.py is for common preprocessing of sentence, e.g. tokenization, creating bigrams, trigrams, unigrams, read and write corpus from disk, etc.

* special features:
  - For problem 3 I leave all the input variables ( including transition matrix, observation matrix, beging and end transition probability, tags and words, testing sentence ) hardedcoded in code rather than interact with users, since there's no sample input file for me to test any automated parsing logic. As a result, you could change the matrix and sentences manully and do your testing. The part of code is in prepare_case_for_ans_3 method of frontend.py
  * If you want to test the functionalities in a segment by segment manner, feel free to change code in the unittest class in each problemm-specific script. A typical name for unittest class would be "testNAME_OF_PROBLEM_CLASS" and it would have "setUp()" and "test_x_NAME_OF_A_METHOD" as the implemented methods. Then simple execute the .py file and it would run the test for you.

## How to set up environment

- All of the code is written in python. As a result, a special python environment has to be created in order ot set up all needed packages on UTD graduate server. To create a python environment, you could use

```
python3 -m venv $PATH_TO_NEW_VIRTUAL_ENVIRONMENT
```

- To install all the neede packages, you could use

```
$PATH_TO_NEW_VIRTUAL_ENVIRONMENT/bin/pip install numpy pandas
```

- Be aware that the requirements.txt is fit for running on UTD graduate server with a python version of 3.5.1 . There's no gurantee on using the requirements.txt file on other machines. If you use pip with a lower version of python3 or on other machines, package version clashes may appear.

## How to run the code

- To run the code, you need to use

```
$PATH_TO_NEW_VIRTUAL_ENVIRONMENT/bin/python frontend.py
```

## How to interact

- Just follow the instructions provided by frontend.py.

## How to see result

- Some of the easy results are available from the console. However, for result with giant output (like the count tables or probability tables). We store the result into an extra csv file that can be visulized via third party tools (like pandas or excel).
