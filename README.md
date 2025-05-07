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

***4 - Feature Engineering***
- How will you handle the original data? Will you use all of it? What will you remove?

- This is the meaty data science part. Having finished step 2, you should have a good idea as to which
features are more useful than others. 
- Now is your time to put that theory to paper and write code that processes the features and feeds them into the model. 
- The aim is to test many different configurations and features so that the model performs its best. 
- You should now be running models on the data and analysing the output. 
- What metrics should you use? Accuracy, Recall, Precision, F1?

- Now is the time to use some pandas magic to load the data and perform operations until you have something you can feed into a model. 
- Read about feature selection. You may generate many new features, how may we choose the best ones? Which are redundant? 
And why do we need to choose at all, instead of using them all? Is more really better? Or on the contrary, does “less is more” fit here?
- How will you handle multiple tables of data?
- In what format will you store the data?

***5 - False Analysis***
- Having settled on the optimal model, run it on your validation set and examine where your model made incorrect predictions. 
- Analyse, comprehensively, why your model produced these results. 
- Can you replicate them? 
- Is it easily fixable? 
- Can you improve your model based on what you learnt?

- For this part you may ask the following questions:
  1. Is my model’s performance perfect? Why? Why not?	
  2. Is my model ‘overfitted’? Over what set(s)?
  3. What inputs tricked my model? How do they reflect in my metric?
  4. What may I change in my model in order to avoid that?
  5. What properties in my model should I inspect to understand its weaknesses?
  6. How could I visualise those properties?

  - And after fixing those patches:
  1. Did my changes fix the problem?
  2. Are there other implications to my changes?
  3. Is the patched model stronger or did we hurt its computational expressibility?
  4. Had I had more time, what changes to the model would I have made?


***6 - Testing the Model***
- Now we have finished the final adjustments on the model. How does it perform on the test set?
- Run the model on the test set. How does it perform?
- Examining the results, are they as you expected?
- Reflect. Why did your model perform as it did on the test set?

- Congratulations, you have finished the exercise! Best of luck for what lies ahead in your data science journey. 
