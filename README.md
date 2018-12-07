## DOCUMENT CLUSTERING

### INTRODUCTION

Before applying predictive modeling to a large volume of unstructured text, we need to evaluate the quality of data. Instead of evaluating every single document in the large amount of data, we can cluster similar documents together and evaluate the most representative document within the cluster of documents. Additionally, document clustering can also disclose the nature of numerical vectors converted from unstructured text to see how good the data is before feeding into any further supervised learning model.

In order to deal with unstructured text, we need the word embedding model to convert them into numeric vectors

#### INSTALLATION:
Activate your virtualenv
pip install -r requirements.txt

### BUILDING AND EVALUATING WORD EMBEDDING MODEL

1. Prepare Training Dataset:

- All training dataset has been merged into *merged.txt* file in the corpus directory
- If you want to prepare your own training dataset, zip all your text file and place in the corpus directory
- Run mergefile.py: python mergefile.py

2. Train word embeddings:

   ```
   python train.py --size=300 --epochs=100 --corpus=filename
   ```

   * Argument explanation:

   size : vector size of your output , default = 300

   epochs: number of epochs, default = 100

   corpus: corpus text file, you can specify your own corpus text file or use default text file *corpus/merged.txt*

   * As an outcome, there are 4 word embedding models:
     *  Word2Vec model without bigram to detect phrases (built by using Gensim)
     *  Word2vec model with bigram (built by using Gensim)
     *  Word2Vec model with Port stemming
     *  Doc2Vec model with bigram (built by using Gensim)
     *  Doc2Vec model with Port stemming
     *  Glove model with bigram (built by using glove-python)

3. Evaluate word embeddings:

   There are two Model-Evaluating scripts: 

   a) __semantic_evaluate.py__ : used to evaluate how good the models are at maintaining the semantic similarity among medical terms

   ```
   python semantic_evaluate.py
   ```

   **Task 1**: Qualitative Evaluation

   **Data Description** 

   There are 8 target words belonging to 3 different categories: disorder, symptom and drug.

   We evaluate the word embedding models by finding the top 5 most similar words pertaining to 8 target words

   **Output**

   The output is at *data/intrinsic_eval.csv* which shows a list of top 5 most similar words

   Result: Word2Vec with phrases detection gives the best result

   **Task 2**:  Quantitative Evaluation:

   There are pairs of medical terms with scores judged by doctors, clinic coders.

   **Data Description**:

    *data/UMNSRS_rel.csv*: shows the relatedness scores between pairs of medical terms

    *data/UMNSRS_sim.csv*: shows the similarity scores between pairs of medical terms

    Link to dataset: http://www.people.vcu.edu/~btmcinnes/data.html#mayosrs

    **Output**:

    The correlation coefficient to show how the expert-judging scores are co-related with the Model-producing score. The score is in the range from -1 to 1 where 1 indicates our model perfectly produces output as expert's expectation.

   Word2Vec has the best performance with coefficient values > 0.6

   Among all techniques we apply on Word2Vec, Word2Vec with bigram to detect phrases performs better than Word2Vec without phrase detection. When applying Port stemming, we can capture the true meaning of words which significantly improves the Word Embedding Model.

   W2V using both bigram to detect phrases and porter stemming produces the coefficient value of 0.635124.

   b) __supervised_evaluate.py__: used to evaluate how intelligent the model is at capturing the core information of any unstructured text 

   **Data Description**:

   *data/drlounge_all.csv*: data scraped from drlounge forum including three columns: question raised by patient, answer by doctor and the label column which is the disease related to the Question-Answer

   *data/i2b2_smoker_status.csv*: dataset from i2b2 which includes the discharge summary about patients and their smoking status as label column.

   **Task Description**:

   Our task is to use the word embedding model to convert those unstructured texts into numerical vectors and use those vectors to predict the label. If the word embedding model is of a good quality, it needs to produce valuable numerical vectors which maintain the insight information within texts. Namely, when using vectors for the classification task, the accuracy needs to be high



    Running
    
    ```
    python supervised_evaluate.py
    ```



    **Output**:

   The accuracy when using numerical vectors to classify texts from drlounge forum and i2b2.

   For i2b2 dataset, word2vec with stem and without stem give the best accuracy with 65%.

   For drlounge dataset, word2vec without stem gives the best accuracy with 54%.



### Clustering:

1. Data Exploration:

Self-Organizing Maps are very useful to Big Data. They can be used to

* reduce the dimensionality of a learning problem, and
* obtain a deeper insight into the problem domain, and
* visualize high-dimensional data, and
* serve as a pre-processor for cluster analysis, and
* enrich data through augmentation for further processing

Explanation: Self-Organizing Map is a type of ANN which uses a neighborhood function to preserve the topological properties of the input space.

**Result**:

Heatmap:

![heatmap](https://github.com/StevePhan97/DocClustering/blob/master/assets/heatmap.png)

The light area indicates the high density of input data points got mapped into our 2D Map. As we can see, the data distribution is random. The district of data points with greatest density is located in the upper left region which potentially indicates a separated cluster with an irregular shape.

U-Matrix:

![u_matrix](https://github.com/StevePhan97/DocClustering/blob/master/assets/u_matrix.png)

The strong color indicates that the coordinates on 2D map which represent input data points near that location in the input topology are distant from the other coordinates.

At the lower right region of the 2D map, the highly distant group of input data points are spotted. These data points may potentially be the outliers.

1. KMeans:

With a lot of outliers, random distribution and irregular shape clusters, KMeans cannot perform well on this data. We first plot the graph of the number of clusters k vs the squared error from any points to its centroid and observe the pattern when increasing the number of clusters:

![kmeans_error](https://github.com/StevePhan97/DocClustering/blob/master/assets/kmeans_error.png)

As expected, the 'right' number of clusters k does not exist as the error keeps decreasing monotonically when k increases. We could not observe the 'knee' where the error rate becomes small as k increases, the curve becomes flatten out at some k value. In other words,the best cost-benefit is observed.

3. HDBSCAN:

When contemplating the heatmap, we can clearly conclude that the clusters may come with varying density which critically affects the performance of DBSCAN.

HDBSCAN, on the other hand, extend DBSCAN by converting it to hierarchical clustering. HDBSCAN uses mutual reachability distance which considers the distance to k-th nearest neighbor. Namely, HDBSCAN observes the context of a given point and chooses different threshold for neighborhood radius to 'partition' the clusters of varying density.

Running Script:

```
python clustering.py
```

Results:

![umatrix_hdbscan](https://github.com/StevePhan97/DocClustering/blob/master/assets/umatrix_hdbscan.png)

From the HDBSCAN results, HDBSCAN successfully identifies a group of data points with high density in the upper left region on 2D map. There are two clusters of irregular shapes identified by HDBSCAN. The red clusters are separated on the map which can be explained by the fact that Self-Organizing Map loses some 'configuration information' when mapping 350-dimension data into 2-dimension map. Another reason contributes to the learning process of Self-Organizing Map when there are too many outliers mingling within the potential clusters, and Self-Organizing Map could find the way to merge them together on 2D map.

### Conclusion
* Stemming and detecting phrases before feeding documents into Word2Vec model will improve the quality of output vectors since the core information within texts are maintained and conveyed effectively in the numerical vectors.

* The unstructured texts from DrLounge forums are difficult to be clustered possibly because patients sometimes describe their symptoms in vague, or there are a lot of infrequent medical terms which are supposed to contain important information, but inversely, introduce outliers to our unsupervised learning process. As a suggestion, we should use our domain knowledge to detect special medical terms and replace them to increase the word frequencies while maintaining the original meanings as much as possible.

* HDBSCAN proves most effective in our case and successfully identifies two clusters which are intuitively based on the assumptions we made in data exploration phase using Self-Organizing Map.

* In order to improve the clustering results, we in fact need a larger corpus for the Word Embedding Model which can be difficult to achieve due to privacy issues 
