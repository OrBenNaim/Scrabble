***Scrabble - שבץ נא***
- After finishing the introductory data science course, we will use our newfound skills and knowledge to rank players based on a game of scrabble.

***Introduction***
- The Scrabble dataset is a medium-sized dataset which contains the games, turn, and ranking data collected from Woogles.io gameplay. Three bots played ~73,000 games at different difficulties. The games were played against registered users. The aim of the exercise is to use the game metadata to predict the ranking of the registered users. 

- Here is an example of a game played by on Woogles.io: https://woogles.io/game/icNJtmxy 

***Exercise Outline***
- Your aim is to build the best Machine Learning model possible to predict the ranking of the registered players. 
- We have built an outline for you to follow—this will help you design, build,
  and test models and introduce you to designing machine learning models and pipelines. 
Setup
Exploratory Data Analysis (EDA)
Model and Pipeline Development
Feature Engineering
False Analysis
Testing the Model


***1 - Setup***
- Download the data. Link here to 
Set up your working environment (Python interpreter, venv, Jupyter notebook, etc.). 
- Consider building some base code that will help you expand your project later on. Think ahead!
Manually look through the data, make note of any interesting features and tendencies. 
Remember, this is just the setup and shouldn’t take too long. 

***2 - Exploratory Data Analysis (EDA)***
- Exploratory Data Analysis (or EDA) means to summarize the main characteristics of the data. 
- This is often done using statistical graphs and other visualization methods. E.g., histograms/distributions, correlations between features, etc. 
- Is there missing data? How will you cope with this?
- Categorical versus Numerical data, 
- There are seemingly infinite features to measure, analyse, and graph. Consider which features of the data will be most useful. 
Remember, when you are working on a project, it is often more desirable to have a good solution in a short amount of time, than an amazing solution that is overdue.


***3 - Model and Pipeline Development***
- This section is for you to design and build a robust pipeline that receives data and feeds it into a model. 
- You will design the pipeline, take the data, perform a machine learning algorithm, 
and output a model that can be used to make predictions (and test its metrics). 
- This part can often get quite confusing and disorganized, quickly. 
- If you designed the code well in the setup section, it should be easier to organize your experiments. 
- Keep track of what you try, what works, what doesn’t.
- The aim is to build a pipeline that will allow you to keep track of your model and changes you will make. 

- Consider asking (and answering) the following questions:
  - How do we tune hyperparameters?
  - Which model to choose? How do they handle categorical features?
  - What metric do we use when we say a model is ‘better’?
  - Read about metrics: the following blog post offers a broad review, although you may read additional materials. Note to yourself what are the strengths and weaknesses of each metric. Which ones fit your problem?
  - After choosing the best model, you should have gained a final understanding of:
  - The model’s performance on its train set
  - The model’s performance on its validation set
- Remember to test your model only on the training data. The test data is for final testing only.
