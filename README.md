# Module 19: Neural Networks and Deep Learning Models

## Overview of the Analysis

### Purpose
The purpose of this analysis was to create a binary classifier that would be capable of predicting if applicants would be successful if funded by Alphabet Soup. The dataset contained 34,000 organizations that have recieved funding from Alphabet Soup. In this analysis, the data was preprocessed, compiled, trained, and evaluated using a neural network model. Finally, the model was optimized to achieve a predictive accuracy of 75%.

### Resources
* Jupyter Notebook, Python 3.7.13
* Python Libraries: scikit-learn, tensorflow, pandas os
* Data Source: [charity_data.csv](https://github.com/daniel-sh-au/UofT_DataBC_Module19_Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
* Challenge Code: [AlphabetSoupCharity.ipynb](https://github.com/daniel-sh-au/UofT_DataBC_Module19_Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb), [AlphabetSoupCharity_Optimization.ipynb](https://github.com/daniel-sh-au/UofT_DataBC_Module19_Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)

## Results

### Data Preprocessing
* The **IS_SUCCESSFUL** column was considered the target for the model
* The features of the model included all columns in the dataset except for the IS_SUCCESSFUL column and the dropped columns:
    * APPLICATION_TYPE
    * AFFILIATION
    * CLASSIFICATION
    * USE_CASE
    * ORGANIZATION
    * STATUS
    * INCOME_AMT
    * SPECIAL_CONSIDERATIONS
    * ASK_AMT
* The **EIN** and **NAME** columns were not considered targets nor features and were removed from the dataset

### Compiling, Training, and Evaluating the Model

* Layers: 3
    * Input Layer
    * First Hidden Layer
    * Output Layer
* Neurons: 
    * Input Layer: 80
    * First Hidden Layer: 30
* Activation Function
    * Input Layer: ReLu
    * First Hidden Layer: ReLu
    * Output Layer: Sigmoid
* This model was selected to be that base for our neural network analysis. Modifications to the initial model were added to improve the model to an accuracy of 75%. 
* The target model performance goal of an accuracy of 75% was not achieved. From the optimization attempts, the accuracy remained at approximately 72.8%
* The steps to improve the model are shown below: 



## Summary