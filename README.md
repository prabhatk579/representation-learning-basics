<h1 align = center>Representation Learning Basics</h1>

## Requirements
- numpy
- pandas
- matplotlib
- sklearn
- re
- string
- nltk

# Part A - Representation Learning

Part A consists of the following topics:

- Dimensionality Reduction Introduction
- Missing value Ratio
- Low Variance filter
- High Correlation Filter
- Random Forest
- Backward Feature Elemination
- Forward Feaeture Selection
- Singular Value decomposition (SVD)
- Principle Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

Dataset Used is [Fashion MNIST Training Dataset](https://www.kaggle.com/zalando-research/fashionmnist?select=fashion-mnist_train.csv)

## Dimensionality Reduction Introduction

#### 1. a) What is Dimensionality Reduction ?
**Ans.** In a dataset, the number of variables or features present is known as Dimensions.

If data is represented using rows and columns, such as in a spreadsheet, then the input variables are the columns that are fed as input to a model to predict the target variable. Input variables are also called features.

When the dataset is huge, there will be a lot of variables and features, thus it will have higher dimensions.

When we deal with high dimnesional data, then its more chalanging to do predictive modeling on a model,It's also hard for us to visualize the data and it also affects the output of Machine Learning Algorithm as there might be some grabage, irrelevant, missing or repetative data, Which is also known as "Curse of Dimensionality". This increases the cost of the storage of data and uses a huge amount of resources.

Dimensionality reduction may can be defined as the process or technique to remove the 'Uninformative information' while retaining the informational data. In laymen words, it's a process of removing input variable from the database such that the removal of the variables does not change the database output by a huge amount.

#### 1. b) Why is dimensionality reduction required?
**Ans.** Dimensionality reduction is required to remove the 'Uninformative information' from the data which in return will do the following things:

It will reduce the size of the database as dimension of the database is reduced.
Less dimensions lead to less computation and training time
Some algorithms do not perform well when we have a large dimensions. So reducing these dimensions needs to happen for the algorithm to be useful.
It's hard for us to visualize the data if it have higher dimensionality so dimensionality reduction helps us to visualize the data.
After dimensionality reduction, the output of the machine learning model will be more accurate when compared to uncleaned high dimensional dataset.


### Missing Value Ratio
When a dataset is given to us which have huge dimenisons, there is a chance that some values are missing in the dataset.

Here question arrises, If the dataset does not have values associated to the variable so chould be impute the variable or drop the variable.

To decide if we want to impute the variavle or drop the variable we calculate the 'Missing Value Ratio' which will help us to decide we impute or drop the varaiable if the missing value have a threshold of x(say 50%). If the values are more than x then we impute the variable otherwise we drop the variable.

### Low Variance Filter
In our dataset say some variable have same values (say z), if we include that variable then our model would be improved? `Maybe...`

**If the variance of that variable is 0 so it would not be very heplful for us.** So, we remove variables with low or 0 variance.

### High Correlation Filter

High correlation between two variables may can be defined as they have similar trends and are likely to have information similar to each other
This can bring down the performance of some models drastically. We can calculate the correlation between independent numerical variables.

If corelation coefficeint overcomes some threshold value (say 80%) then we will drop one of the two variable.
Since dropping variable is not ideal so we drop the variable taking ddomain into consideration.

`df.corr()` gives us the correlation of each variable

### Random Forest
Random forest may can be defined as an algorithm which is used to do feature selection which helps us select a smaller subset of features.

We can impliment ramdom forest by importing from `sklearn.ensemble` library called `RandomForestRegressor`. Since `RandomForestRegressor` only takes in numerical values thats why we tends to convert the data into the numeric form.

After applying the RandomForestRegressor, we visualize the importance of each variable by plotting it into a graph. `model.feature_importance_` gives us the importance of each feature and we plot the features.

We then use `SelectFromModel` from `sklearn.feature_selection` which decides the importance with respect to their weight.

### Backward Feature Elimination
Backward Feature Elemination is a method to reduce the dimension by following some simple steps:

> Step - 1: First we take whole set from the training set and calculate its performance (say n variable).

> Step - 2: We now remove one of the variable (one at a time) (n-1 variables) and calculate its performance and compare it to the previous performance.

> Step - 3: We remove the variable which increases the performance on removal.

We continue removing variables one by one until it increase our criterion.

### Forward Feature Selection
Forward Feature Selection is a method to reduce the dimension by following some simple steps:

> Step - 1: First we take empty set (containing 0 elements).

> Step - 2: We include one element to the empty set.

> Step - 3: We now include one of the variable (one at a time) (i+1 variables) and calculate its performance and compare it to the previous performance.

> Step - 4: We add the variable which increases the performance on addition.

We continue adding variables one by one until it increase our criterion.

### Singular Value Decomposition (SVD)
SVD or Singual value decomposition is a dimensionality reduction technique which decomposes the maatrix created by the variables and features into three matrices smaller matrices.

### Principal Component Analysis
PCA or principal Component analysis is an unsupervised dimensionality reduction technique in which a principal component is a linear combination of the original variables. These Principal components are extracted in such a way that the first principal component explains maximum variance in the dataset. Second principal component tries to explain the remaining variance in the dataset and is uncorrelated to the first principal component. Third principal component tries to explain the variance which is not explained by the first two principal components and so on.

It can be easily done by importing `PCA` from `skklearn.decomposition` and applying `pca.fit()` on the dataframe.

### Linear Discriminant Analysis
LDA or Linear Discriminant Analysis approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally finds the axes that maximize the separation between multiple classes. 

We can easily apply LDA by importing `LinearDiscriminantAnalysisfrom` the library `sklearn.discriminant_analysis` and applying `sklearn_lda.fit_transform()` over the dataframe.
