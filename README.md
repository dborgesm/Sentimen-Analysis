# Sentiment Analysis

In this project, using the reviews from Kickstarter and IndieGoGo a sentiment analysis was performed to analyze the impact of these sentiments on crowdfunding campaigns. By using neural networks, our main objective is to determine the attitude of the author towards the company. In addition, the performance of each model was evaluated using the accuracy and the area under the curve.

## Data Preprocessing 

The data set contained the sentiment, which determines if the text was positive (1) or negative (0). The confidence, that indicated the strength of the sentiment and the text that contained the reviews. The embeddings used were fasttext and GloVe

**Missing values**
There were no missing values found in the data. 

**Confidence**
This variable had values between 0 and 22 with more values less than 1. It was assumed that the cases with value greater than 1 were human errors. Hence, they were converted to percentage to have all the values between 0 and 1. This indicated that, if the confidence was closer to zero the sentiment was weak and closer to 1 strong. It was found out that the sentences with confidence zero did not capture the sentiment of the reviewer. Thus, the model was only trained with sentences that had confidence.

**Sentiment**

This variable was slightly unbalanced because 30% of the sentiments were negative and 70% were positive, this indicated that people gave more positive reviews.

**Text**

The objective to cleaning the sentences was to make them easier to read by a machine and to pass the significant values to the neural networks. Hence, an overview to the data indicated that the maximum number of words a review had was over 1200 words and as minimum 1. There were some comments with less than 5 sentences that contained random words for example “itai”, “CF”, and so forth. These were removed from the data. 

Another important thing to consider were sentences that had URLs, however only 2.5% of the records included them, which indicated that even keeping only the domain would not be necessary, thus the complete URLs were removed.

This were the three main approaches involved in the data preprocessing.
-	Lowercasing. This helped to eliminate sparsity issues when same words use different cases. This could be fixed if the data were large enough but is not the case.
-	Lemmatization. This method was used to get the root form of the words, for instance instead of “having” the word returned was “have”, it is commonly used for sentiment analysis because the sentence does not lose its meaning.  
-	Removal of stop words and conjunctions. These are referred to the most common use words in the English language, for instance “the”, “a”, “but”, “and”, and so forth. They did not have significant information to contribute to the model. However, pronouns, verbs to be and negative stop words, as not or no, were kept in the data because the sentence could lose its meaning without them. 

The additional steps involved in text preparation were using the methods applied in a [twitter sentiment analysis](https://github.com/charlesmalafosse/FastText-sentiment-analysis-for-tweets). 
-	Removing and replacing contractions, i.e. do not instead of don’t.
-	Removing numbers.
-	Removing repeating characters in a word. 
-	Removing punctuation and multiple spaces
-	Converting the emojis to a label, for instance “:-)” and its combinations were changed to smiley, “:-(” was changed to sad and “:-P” was changed to playful.
-	Removing single characters.
After applying these steps, the average word per sentence was 22.38 and a maximum of 807 words (Figure 4). The top 4 used words were “you”, “is”, “have” and “not”.

**Embedding**

Pre-trained embeddings were used because they can improve the performance of the analysis by mapping the words in a statistically robust manner into an embedding space. The two pretrained embeddings used were the following:

1.	fastText for english trained on Common Crawl and Wikipedia.
2.	GloVe (Common Crawl glove.42B-300d).

The GloVe embedding learned by constructing a co-occurrence that counts the frequency of the appearance of the word.  Meanwhile, fastText uses n-gram characters and the benefit of using it was because it can generate embeddings for words that did not appear in the training set. Both embedding dictionaries contained a maximum length of 88 words because this eliminates at most 5% of our data. Furthermore, pre padding was used because the relevant information was found at the end of the statement. On the other side, because common crawl is trained in some forums, words as “lol”, “lmao” were kept.

## Modelling

### Architecture 1

The first architecture used a Convolutional Neural Network (CNN) based on Yoon Kim’s model. The architecture was the following:

- An embedding layer.
- A Dropout layer with probability 0.6.
- A parallel model of four layers with input shape of 88 (maximum words) by 300 (embedding size), each one consisting of:
  - A Conv1D layer where each looked for 32 different combinations of 3 words. 
  - A max pooling layer of size 3. 
  - A concatenate layer.
 - A flattening layer.
 - Two consecutive dense layers of size 64 and a dropout layer with probability 0.5 after each one of them. 
 - An output layer of size 1 with sigmoid as activation function.
 - The loss used was binary cross entropy and Adam as optimizer.
 
The drop out layer at the beginning helped the validation to not instantly increase over the training
loss (see Appendix B.1) and the greater the probability the aggressive the effect over the losses trends, hence keeping a probability of 0.6 showed better results. In respect of the parallel model 4 layers of the same size showed a better behavior, by changing the model to greater sizes the resulted model was just learning noise. Now regarding, the filter sizes of the
CNN, the model was learning quite fast if the size was increased, using 32 controlled this behavior. In addition, a max pooling of the same size of the filters (3) was better than using  because it also made the validation loss grow fast. The two consecutive dense layers with drop out, were used to feed all outputs from the flatten layer to all its neurons with a size of 64, if the size increased the model was overfitting. The training rate was not change because the models were performing the same even if it was reduced even adding a normalization layer did not improve the model.

### Architecture 2

The second architecture used was combination of a Bidirectional Gated Recurrent Unit (GRU) and CNNs. GRU is a type of recurrent neural network which can remember previous information for longer time by using hidden states and connecting it to the current task, also it keeps the contextual information in both directions.

The following architecture worked better for the fastText embedding.

-	An embedding layer with shape 88 (maximum words) by 300 (embedding size)
-	A Dropout layer with probability 0.5
-	A Bidirectional GRU layer with 128 filters 
-	Two Conv1D layers where each one of them looked for 16 different combinations of 4 words with strides equal to 2.
-	A Global Max Pooling
-	Two consecutive dense layers of size 64 and a dropout layer with probability 0.5 after each one of them.
-	An output layer if size 1 with sigmoid as activation function. 
-	The loss used was binary cross entropy and Adam as optimizer. 

The drop out layer at the beginning, was used for the same reason as mention before, decreasing the probability did not change the model significantly. In respect of the Bidirectional Layer 128 filters, decreasing them made the model overfit quickly. The filters used in the CNNs also showed that by increasing them the model showed worst results, the validation loss grew faster over the training loss. The strides, the more they were increased the less model could learn, hence keeping it at 2 showed that the models learn more at the beginning but stopped learning at some point. On the other side, global max pooling was used because it summarises the strongest activation and it is commonly use in language processing. The two consecutive dense layers with drop out helped us the same way as mentioned in the first architecture. The training rate was not change because the models were performing the same even if it was reduced even adding a normalization layer did not improve the model.

### Activation and Loss Function

The activation function used for hidden layers was Relu because it is the one that models better how the brain works and rectifies the vanishing gradient problem.  For the output layer, sigmoid activation function was used because is a binary classification problem and they are difficult to optimize if it is used as a hidden layer. Finally, the loss function to measure the error of the prediction was binary cross-entropy due to there are only two classes. 

### Models

The data was split 80/20 for training and testing. Fifty epochs were performed in each model, so the performance could be appreciated and they were stopped when the models started to learn noise (training loss decreasing and validation loss was iterating in a horizontal line) or overfit (training loss decreasing and validation loss increasing).

#### FastText using architecture 1

The model started learning a little at the beginning, however after the 16th iteration the model started to learn only noise.

The confusion matrix showed that the model was better at classifying the reviews with positive sentiments and only a few were misclassified. The accuracy was 81.98% and the Area Under the Curve (AUC) was 89.38%.

#### FastText using architecture 2

The first plot showed that the model was learning because both the validation and training set were decreasing, but at iteration 34th it started to learn noise.

The confusion matrix showed that the model was very good at classifying both positive and negative, and only a few were misclassified. The accuracy was 86.90% and the Area Under the Curve (AUC) was 94.19%. Overall, this model was performing better in terms of accuracy in comparison to the first one.

#### Glove using architecture 1

This model learnt a little at the beginning, but it was stopped at iteration 30 because that was where the model started to learn noise.

The confusion matrix showed that the model was better at classifying the reviews with positive sentiments (88.15%) but it was worst a classifying the negative ones (68.88%). The accuracy was 81.83% and the Area Under the Curve (AUC) was 89.38%.

#### Glove using architecture 2

This model was learning significantly before the 30 iteration. 

The confusion matrix showed that the model was better at classifying the reviews with positive sentiments (88.47%) but it was worst a classifying the negative ones (84.15%). The accuracy was 87.06% and the Area Under the Curve (AUC) was 94.19%.

After analyzing the confusion matrix, all the models were better at classifying the reviews with positive sentiment and the accuracy of all the models were greater than 80%. This was due to all the models learnt well at the beginning and they were stop when the performance was decreasing. Also, some of the models were better at classifying true positives than true negatives because of the unbalanced data. 


### AUC-ROC

The embedding that used the Bidirectional GRU with CNNs had better performance in terms of AUC. Although, just by a small percentage the Glove embedding worked better in the data.

## Conclusion

For this data set the embedding that worked best was Glove using the Bidirectional GRU with CNNs. All the changes and modifications to the models were done by observing the behavior of the validation and training loss, however because of the iterating trend of the validation losses I am assuming these models are too simple and must be improved to obtain better performance.

Here are the codes for [Data Preprocessing and EDA](https://github.com/dborgesm/Sentiment-Analysis/blob/master/Data_preprocessing_and_Embeddings.ipynb) and for [Modelling](https://github.com/dborgesm/Sentiment-Analysis/blob/master/Modelling%20Sentiment%20Analysis.ipynb)
