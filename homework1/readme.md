# CS6320 Assignment 1

This is the assignment repo for first assignment for NLP course 22 spring

## How to set up environment

- Most of the part of code is written in python. As a result, a special python environment has to be created in order ot set up all needed packages on UTD graduate server. To create a python environment, you could use

```
python3 -m venv $PATH_TO_NEW_VIRTUAL_ENVIRONMENT
```

- To install all the neede packages, you could use

```
$PATH_TO_NEW_VIRTUAL_ENVIRONMENT/bin/pip install -r requirements.txt
```

- Be aware that the requirements.txt is fit for running on UTD graduate server with a python version of 3.5.1 . There's no gurantee on using the requirements.txt file on other machines. If you use pip with a lower version of python3 or on other machines, package version problems may appear.

## How to run the code

- To run the code, you need to use

```
$PATH_TO_NEW_VIRTUAL_ENVIRONMENT/bin/python frontend.py
```

## How to interact

- Just follow the instructions provided by frontend.py.

## How to see result

- Some of the easy results are available from the console. However, for result with a giant output (like the count tables or probability tables). We store the result into an extra csv file that can be visulized via third party tools (like pandas or excel).
