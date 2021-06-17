<h1 align = center>Representation Learning Basics</h1>

## Requirements
- numpy
- pandas
- matplotlib
- sklearn
- re
- string
- nltk

# Part-A: Representation Learning

Part-A consists of the following topics:

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

# Part-B: Feature Extraction in Text

Part-B consists of following topics:

- Manual tokenization
- Tokenization and Cleaning with NLTK
- Word Counts with CountVectorizer
- TF(Term Frequency) - IDF(Inverse Document Frequency)
- Word Frequencies with TfidfVectorizer
- The Bag-of-Word Model

## Feature Extraction in Text
As we know that the machine learning algoriths need numbers to work with. As rax text is messy thats why machines can not work directly on the texts. So we convert or process th text into a format that machine can understand. The process of conversion or the cleaning is called feature extraction in texts.

### Manual Tokenization
#### Loadinig the data
We load the data using the `open()` function including its address and what we need to do with the file. Here we use '`r+`' which is read and write.
At last we `close()` the file to save the changes in the file.

#### Spliting the data
Text file can be vey messy without a proper format. So to bring uniformity we use whitespaces to seperate the words from space, newlines, tabs, etc.
We can do this in Python with the `split()` function on the loaded string.

Alternative approach is to use regex model (re) and split the document into words by selecting for strings of alphanumeric characters. firstly we `import re`; Then we use the function `re.split` to split the data.

#### Split and Removal of punctuations
we want words without punctuations (like ',' , '.' , '$' ,etc) so we import punctuations from string and use ragex for character filtering. The function `sub()` replaces the punctuation with nothing.

#### Normalizing case
We can normalize the case by calling `lower()` function to each word.

### Tokenization and Cleaning with NLTK
The Natural Language Toolkit, or NLTK for short, is a Python library written for working and modeling text. We import the nltk library and download punkt and stopwords which contais the puunctuations and the stopwords

#### Spliting into the Sentences
NLTK provides the `sent_tokenize()` function to split text into sentences. It will split paragraphs into the sentences.

#### Split into Words
NLTK provides a function called `word_tokenize()` for splitting strings into tokens and we normalize the case for every word. 

#### Filter out the Punctuations.
Punctioations are those that differentiate between the ending of a sentence or for spacing. The most common puuncuations are: ' __.__ ', ' __,__ ', ' __:__ ', ' __;__ ', ' __"__ ', ' __-__ ', etc.

#### Filter out Stop Words (and Pipeline)
Stop words are those words that do not contribute to the deeper meaning of the phrase.
They are the most common words such as: the, a, is, etc.

#### Stem Words
Words like 'Finding' , 'found' , 'finds' , etc all reduces to stem 'find'.

We can easily filter stem words by using NLTK via the `PorterStemmer` class.

### Word Counts with `CountVectorizer`
The `CountVectorizer` provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.

You can use it as follows:
- Create an instance of the CountVectorizer class.
- Call the `fit()` function in order to learn a vocabulary from one or more documents.
- Call the `transform()` function on one or more documents as needed to encode each as a vector.

An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. Because these vectors will contain a lot of zeros, we call them sparse.

Python provides an efficient way of handling sparse vectors in the `scipy.sparse()` package.

The vectors returned from a call to `transform()` will be sparse vectors, and you can transform them back to NumPy arrays to look and better understand what is going on by calling the `toarray()` function.

### TF(Term Frequency) - IDF(Inverse Document Frequency)
TF or Term Frequency may can be defined as a measure of how frequently a term occurs in a document.

##### Formula

> TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)

IDF or Inverse Document Frequency may can be defined as a measure of how important a term is.

##### Formula

> IDF(t) = ln(Total number of documents / Number of documents with term t in it)__*__
<h5 align = right>*Where, ln: log to the base e</h5>

### Word Frequencies with `TfidfVectorizer`
Words like 'The' will appear many times and their large counts will not be very meaningful in the encoded vectors.

An alternative is to calculate word frequencies, and by far the most popular method is called TF-IDF. This is an abriviation that stands for **Term Frequency - Inverse Document Frequency** which are the components of the resulting scores assigned to each word.

- **Term Frequency -** This summarizes how often a given word appears within a document.
- **Inverse Document Frequency -** This downscales words that appear a lot across documents.

The `TfidfVectorizer` will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents.

### The Bag-of-Word Model
The Bag of Words (BoW) model is the simplest form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).

Let’s take three types of movie reviews we saw earlier:

> Review 1: This movie is very scary and long

> Review 2: This movie is not scary and is slow

> Review 3: This movie is spooky and good

We will first build a vocabulary from all the unique words in the above three reviews.

The vocabulary consists of these 11 words: `‘This’, ‘movie’, ‘is’, ‘very’, ‘scary’, ‘and’, ‘long’, ‘not’, ‘slow’, ‘spooky’, ‘good’.`

We can now take each of these words and mark their occurrence in the three movie reviews above with 1s and 0s. This will give us 3 vectors for 3 reviews:

> Vector of Review 1: [1 1 1 1 1 1 1 0 0 0 0]

> Vector of Review 2: [1 1 2 0 0 1 1 0 1 0 0]

> Vector of Review 3: [1 1 1 0 0 0 1 0 0 1 1]

#### Drawbacks of using a Bag-of-Words (BoW) Model:
In the above example, we can have vectors of length 11. However, we start facing issues when we come across new sentences:

1. If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of the vectors would increase too.
2. Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid)
3. We are retaining no information on the grammar of the sentences nor on the ordering of the words in the text.
