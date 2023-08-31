# NJU_NLP_2023
This is the assignment of introductory NLP course in `NJU` 2023. 

> Given an IMDB dataset containing 50k movie reviews, train a model to achieve polar(positive or negative) sentiment analysis: to judge a review is positive or negative.
>
> Specification:key:
>
> + Use LSTM
> + Choose relatively optimal hyperparameters
> + *Optional*:star: Use stop words

+ Dataset :link:[Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) 

  *Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis.](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).*

## Data Preparation 

+ To train the expected classifier, we need to convert `row data (text)` to `vectors` first. We choose **Word2Vec** as this implementation.

+ In the attempt to achieve better results, I choose to use `Brown Corpus` from `NLTK data` to **pre-train** a model first, then **fine-tune** it with comment text.

+ Then we have the tool to convert a comment text into a vector to compute. So go ahead and build our own **Dataset** class to access our data.

+ Besides, divide the data into `training set` (70%), `validation set` (10%) and `test set` (20%).

  > Note:warning:: Because I'm pre-training the **word2vec** model using `Brown Corpus`. Thus some rarely or specially used words may not exist in the dictionary of our model. To fix this problem, I decide to skip these rarely used words instead of assigning a zero vector to it. The reason is that even rarely used words are used to train, the model cannot learn its meaning well due to the limited size of sample, and manually assigning a value to it may interfere the progress of learning. :thinking:Afterall no one knows what zero vector exactly means!

  > Note:warning:: If you choose a large number (like 100 or above) for your word2vec model, don't attempt to save the converted data, which is tremendously huge. Of course you can go and have a try if you don't mind having an exploding PC. PS::wink: Never ask me how I know that. 

  > It's extremely costly to pad all sequences to the max length, so I refer to [this blog](https://blog.csdn.net/Delusional/article/details/113357449). Set a hyperparameter <a name="len_of_seq">`length_of_seq`</a> to represent the length of every sequence. A sequence will be truncated if longer than `length_of_seq`, be padded otherwise.</a>

## Training Process

+ Build a **LSTM** network using `pytorch` as the **encoder**

  + Define our encoder as `nn.LSTM(embed_size, num_hiddens, num_layers)`

  + Define our decoder as `nn.Linear(2 * num_hiddens, 2)`  

    > Here to multiply 2 is because that I concatenate the first and the last hidden state to have a better overall understanding of the sequence. :v:

+ Using a **Linear** unit directly from `nn` as the **decoder**

+ :scroll:List of hyperparameters

  > `embed_size` 	the length of word vectors *(embeddings)*
  >
  > `num_hiddens` 	the number of features of hidden state
  >
  > `num_layers`     the number of hidden layers
  >
  > `batch_size`    the number of samples in each batch
  >
  > [`length_of_seq`](#len_of_seq)   the length of sequence 
  >
  > `loss_fn`     the loss function
  >
  > `optimizer `    the optimizer function
  >
  > `lr`     learning rate
  >
  > `epochs`    the epochs to train *(iterating times)*

+ I select `num_hiddens`, `batch_size`, `loss_fn`, `lr`, `epochs`  as the hyperparameters to iterate through, my **hyperparameter space** are as follows:

```python
    hyperparameters = {
        'num_hiddens': [10, 50, 100],
        'batch_size': [64, 256],
        'lr': [0.01, 1],
        'epochs': [10, 20],
    }
```

+ :scroll:The default values for **not iterated hyperparameters**

  > `embed_size`: `20`
  >
  > `num_layers`: 1
  >
  > `length_of_seq`: `20`
  >
  > `loss_fn`: `nn.CrossEntropyLoss()`
  >
  > `optimizer`: `torch.optim.Adam()`

## Results

> I'm still learning and trying to improving:smiley:, so I can't submit it prefect. Following are some known bugs:
>
> :warning: I get some bugs to be fixed here, in some cases, the loss is calculated to be `NaN`. This is due to the fact that I don't understand the underlying **principles of neural networks** .
>
> :warning: Another bug remains to be fixed is that it doesn't print out the best model trained, 
>
> ​	I figure it out from training curves. This is due to the fact that I don't understand the **scope in python**.

+ :scroll:The best model I trained from above selected hyperparameter is specified as follows:

  > `	num_hiddens`: `50`
  >
  > `batch_size`: `64`
  >
  > `lr`: `0.01`
  >
  > `epochs`: `20`

+  Also, I use `matplotlib.pyplot` to draw some graphs to show how the accuracy and loss vary alongside epochs. You can find these graphs and corresponding hyperparameters in `/graphs` directory

+ The length reports is as follows:

  >Max length: 2494 	Line number: 31481 
  >Min length: 6 	Line number: 27521 
  >Total length: 11711285
  >
  >Average length: 234

## File information

+ `/raw` directory is the raw data and length report
  + `IMDB_dataset.txt` is the initial data
  + `comments.txt` `labels` are separated from `IMDB_dataset.txt`
  + `leng_report.txt` is the length information about comments in this dataset
+ `src` directory is the python source code
  + `separate.py` is used to separate raw data into `comments` and `labels` and to generate `length report`.
  + `word2vec.py` is to train the `word2vec` model
  + `split.py` is to split the whole dataset into `train_set` `test_set` `validation_set`
  + `main.py` defines the `neural network`, DIY `Dataset`, tune the `hyperparameters` and generate `training curves`
+ `/models` directory stores trained `word2vec` model
+ `/data` directory stores the split data
+ `/graphs` directory stores the `training curves` of each specification of hyperparameters
