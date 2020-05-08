# Chaining and the Growth of Linguistic Categories

This repository includes code and data for analyses in the following work: 

`Habibi, A.A., Kemp, C., and Xu, Y. (to appear) Chaining and the growth of linguistic categories. Cognition.`



## Python Files

* [`main.py`](main.py) runs the whole process, using the other files. You can follow the flow of the code from this file, if you need to.
* [`paths.py`](paths.py) indicates the paths to the data and directories used throughout the code.
* [`model_saver.py`](model_saver.py) creates the necessary format for the word embedding input file.
* [`super_words_builder.py`](super_words_builder.py) creates the file that contains the mapping of the classifiers and their nouns for each year with the right format for the input of the program.
* [`the_predictor.py`](the_predictor.py) has the SuperPredictor class that does the prediction for a set of classifier nouns for a number of steps and saves a prediction file.
* [`predict.py`](predict.py) includes the function for running the prediction processes and kernel-width optimization using the SuperPredictor class.
* [`the_analyser.py`](the_analyser.py) contains the SuperPredictionAnalyser class that has the necessary methods to analyse the accuracy of a prediction file.
* [`analyse.py`](analyse.py) includes functions that use the SuperPredictionAnalyser class to do the evaluation and create plots.
* [`simulation_analysis.py`](simulation_analysis.py) creates a simulation of the important chaining mechanisms on a randomly generated data and plots the results.
* [`utilitarian.py`](utilitarian.py) contains some utility functions and classes.


## Data Files 

* [`data/super_words-chi-Luis-YbyY(w2v-en).pkl`](data/super_words-chi-Luis-YbyY(w2v-en).pkl) is the file containing the classifier-noun data over time with the specific format that is used in the code.
* [`data/w2v-chi-en-yby.pkl`](data/w2v-chi-en-yby.pkl) is the word embedding file that maps Chinese nouns to the embedding of their English translations.
* [`data/gwc2016_classifiers/`](data/gwc2016_classifiers/) includes the data on classifier-noun pairs provided by: 
`Morgado da Costa, L., Bond, F., & Gao, H. (2016). Mapping and generating classifiers using an open chineseontology. InProceedings of the 8th Global WordNet Conference.`


* [`English_translation_survey_results.zip`](English_translation_survey_results.zip) includes three xlsx files for surveying 100 sampled Chinese nouns in English translations.