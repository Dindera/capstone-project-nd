# Machine Learning Engineer Nanodegree
## Capstone Proposal
Chidera Ozoh  
March 17th, 2021

## Proposal


### Domain Background


With the increase in the use of social media applications such as Twitter, users have found it as a means of communicating to the world information relating to emergencies, disaster and life threatning events. This information passed through tweets are often difficult to identify since one user's choice of words are different from another user's. With the interest in monitoring these tweets by news agencies and organisations to know when a disaster starts to occur, it is important to detect which particular tweet is announcing this disaster.

Since this is a natural language processing (NLP) related problem, this project will apply necessary methods in processing, visualizing and building a model which predicts if a tweet is real disaster or not. Various researches has been carried out using Tweets data in the past. One particular is the classification for authorship of tweets research carried out by Aborisade and Anwar (2018). They applied and compared classifier methods Naive Bayes and Logistic Regression to determine which model performed better in detecting fake or real news. Before building the model, they carried out pre-processing techniques to produce a quality training data for the model. Some of the techniques include data collection, processing such removing unwanted data, splitting the data. This project is an interesting topic for me because it can also be applied in other challenges of social media such as Fake news detection, and in spam email detection.      
  

### Problem Statement

In this project, the issue of identifying whether a tweet is announcing a disaster will be investigated. This probelem will be solved using Recurrent Neural Network (RNN) model applied to the data which contains processed words. The model will be evaluated and improved if performance is below expectation. 

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The dataset to be used for this project contains alreeady split train and test datasets of 7503 & 3243 unique values repectively. The train dataset contains 4 inputs and one output. Since this dataset is a kaggle competion, no output field is included in the test dataset. The inputs and output are defined below.

_id_: a unique identifier for each tweet.
_text_: text of each tweet.
_location_: where the tweet was sent from.
_keyword_: a particular keyword from the tweet.
_target_: the output that denotes if a tweet is about real disaster(1) or not (0)

The datasets are collected from kaggle and was created by the company figure-eight. The inputs id, text and output will be used for building the model. For visualization, all the inputs and output will be applied accordingly.  

### Solution Statement
_(approx. 1 paragraph)_

The proposed solution for this problem is to process the dataset by cleaning, exploring and preparing for modelling. To clean the data, unwanted/incomplete records will be removed and in the texts, stop words will be removed and bag-of-words applied to determine count of words in each text. The data will be visualised to display charts/plots of keywords, and or locations, with targets.  After that, word dictionary will be generated to identify highest and lowest occuring words. The train dataset will be split to 70% train and 30% evaluation sets. The train set will then be passed to the LSTM model which basically is a recurrent network that uses backpropagation through time to address gradient vanishing problem of RNN. The LSTM model is then passed to a Linear model for the final output.

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

The benchmark model for this project is the Linear model which classifies the tweets into 1 (announcing disaster) and 0 (not announcing disaster). Using Pytorch framework, the model will be built using network of neuron layers.  The model will produce classified results of 1 and 0. This model will be measure by finding the accuracy of the predicted values which is  

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
