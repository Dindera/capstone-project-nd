# Machine Learning Engineer Nanodegree
## Capstone Proposal
Chidera Ozoh  
March 17th, 2021

## Proposal
------------------

### Domain Background
----------------

With the increase in the use of social media applications such as Twitter, users have found it as a means of communicating to the world information relating to emergencies, disaster and life threatning events. This information passed through tweets are often difficult to identify since one user's choice of words are different from another user's. With the interest in monitoring these tweets by news agencies and organisations to know when a disaster starts to occur, it is important to detect which particular tweet is announcing this disaster.

Since this is a [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) (NLP) related problem, this project will apply necessary methods in processing, visualizing and building a model which predicts if a tweet is real disaster or not. Various researches have been carried out using Tweets data in the past. One particular is the classification for authorship of tweets research carried out by [Aborisade and Anwar (2018)](https://ieeexplore.ieee.org/document/8424720). They applied and compared classifier methods Naive Bayes and Logistic Regression to determine which model performed better in detecting fake or real news. Before building the model, they carried out pre-processing techniques to produce a quality training data for the model. Some of the techniques include data collection and cleaning, processing such as removing unwanted records, splitting the data, etc. This project is an interesting topic for me because it can also be applied in other challenges of social media such as Fake news detection, and in spam email detection.      
  

### Problem Statement
--------------------

In this project, the issue of identifying whether a tweet is announcing a disaster will be investigated. This probelem will be solved using Long Short Term Memory (LSTM) layers, a  Recurrent Neural Network (RNN) model applied to the data which contains processed words. The LSTM model will then be passed to a Linear Layer model for final prediction of the output. The Linear model will be evaluated and improved if performance is below expectation. 

### Datasets and Inputs
------------------------

The dataset to be used for this project contains alreeady split train and test datasets of 7503 & 3243 unique values repectively. The train dataset contains 4 inputs and one output. Since this dataset is a [kaggle competion](https://www.kaggle.com/c/nlp-getting-started/data), no output field is included in the test dataset. The inputs and output are defined below.

_id_: a unique identifier for each tweet.
_text_: text of each tweet.
_location_: where the tweet was sent from.
_keyword_: a particular keyword from the tweet.
_target_: the output that denotes if a tweet is about real disaster(1) or not (0)

The datasets are collected from kaggle and was created by the company [figure-eight](https://www.figure-eight.com/data-for-everyone/). The inputs id, text and output will be used for building the model. For visualization, all the inputs and output will be applied accordingly.  

### Solution Statement
----------------------

The proposed solution for this problem is to process the dataset by cleaning, exploring and preparing for modelling. To clean the data, unwanted/incomplete records will be removed and in the texts, stop words will be removed and bag-of-words applied to determine count of words in each text. The data will be visualised to display charts/plots of keywords, and or locations, with targets.  After that, word dictionary will be generated to identify highest and lowest occuring words. The train dataset will be split to 70% train and 30% evaluation sets. The train set will then be passed to the LSTM model which basically is a recurrent network that uses backpropagation through time to address gradient vanishing problem of RNN. The LSTM model is then passed to the Linear model for the final output. Being a classifier algorithm, the Linear Layer model is choosen for this project because the input feature is passed through series of network layers and trained severally using epochs to get the best output and minimum loss. The predicted output is calculated thus. _y_ = x.w + b. Where x is the input features, w is weight and b is the bias added to the input.


### Benchmark Model
-------------------------

The benchmark model for this project is the Linear model which classifies the tweets into 1 (announcing disaster) and 0 (not announcing disaster). Using Pytorch framework, the model will be built using network of neuron layers.  The model will produce classified results of 1 and 0. Then the model will be evaluated by measuring the accuracy of the predicted values. The accuracy score can be determined by the number of false positive, false negatives, true positives and true negatives. The area under curve (AUC) metric will also be calculated to use the true positive rate, true negative rate and false positive rate and plot the AUC.   


### Evaluation Metrics
---------------------------

The metrics for measuring the performance of the model includes accuracy and AUC. 
The accuracy can be calculated by

  accuracy = number of correct predictions / total number of predictions

The AUC is determined by plotting the values of true positive rate (TPR), true negative rate (TNR) and false positive rate (FPR). They are calculated thus.

_TPR_ = TP / FN + TP
_TNR_ = TN / TN + FP
_FPR_ = FP / TN + FP

Where TP is true positives, FN is false negatives, TN is true negatives and FP is false positives. 
All these are derived from the predictions and will determine if the model will rank a randomly chosen positive value higher than a randomly chosen negative value.  

### Project Design
---------------------

The design of this project follows the steps listed below and explained briefly afterward.

_Collection / downloading of data_
_Cleaning and preparing of data_
_Visual analysis of data_
_Preprocessing data for modelling_
_Modelling processed data_
_Testing and Evaluation of results_


**Collection / downloading of data**

As stated before, the data will be collected from [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data) and will be stored in the project root directory. 

**Cleaning and preparing of data**

Afterwards, the data will be cleaned by checking if there are empty records and deciding whether to remove or add default data. This stage prepares the data for visual analysis of the data.

**Visual analysis of data**

Using bar charts and scatter plots, the location and keyword input features will be displayed to visualize their values and how they relate. The output data will also be plotted in a bar chart to show the quantity of 1 to 0 values and determine if this will affect the model performance.

**Preprocessing data for modelling**

For this stage, the text input feature will be the focus. The texts will be processed by removing for instance tags, unknown characters,  uwanted words (stop-words), converting the words to small letter and stemming the words so that letters such as _artificial_ and _artificially_ are one word. The stemmed words are then counted (bag of words) and tokenized to give the final data for modelling. The data will be uploaded to Amazon s3 bucket after processing and splitting the train data into train and evaluation sets.    

**Modelling process data**

This stage will be done using custom Pytorch model. Hence I will write functions to train and predict 
the model, and functions to build the model. The train function will be reponsible for downloading the data, processing the data, and applying the model function to a Pytorch estimator to execute the training task. Finally, the predict function will use the trained model to predict the evaluation data.

**Testing and Evaluation of results**

Using the predicted values, the accuracy score and area of curve value will be determined. If the performance is poor, the training and model parameters such as epochs, hidden layers and optimizer will be changed to attain the desired results. 

To summarize, some of the tools/libraries that will aid in implementing this project will include Python-conda environment, Amazon sagemaker, Jupyter notebook, Pandas, Numpy, Sklearn, Pytorch, MatPlotLib etc. The results of the project will be documented in a review report.  

-----------
