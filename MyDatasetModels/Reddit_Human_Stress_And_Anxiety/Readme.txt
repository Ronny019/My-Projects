The attached sample datasetcontains text data posted on subreddits related to human stress and anxiety. 
This dataset contains various issuesrelated to mental health shared by people publicly.
What are the top sources of stress or mental health issuesmentioned in the subreddits posts? 


The usual strategy for NLP is to encode the words into numeric values. However, not every word is meaningful and therefore we have to eliminate those.
Stopwords are a good start.
In this assignment I have used the TF-IDF(Term Frequency Inverse Document Frequency) technique to
assign weights to the words. The way the TF-IDF technique works is it looks for the frequency of the
token in the document and across documents. If a term is used in multiple documents that means the
term is not very specific to a single topic. If a term is used often within a single document it means that
the term carries weight. Adjusting for this two factors every token is assigned a weight.
Term for which the TF-IDF value is above a certain threshold (0.4 in case of this assignment) are
considered important terms. While performing the TF-IDF process the API was also instructed to only
consider those words which are of length 4 or larger. Words which are shorter in length will probably
not be keywords which we are looking for.
After TF-IDF is applied we get a sparse matrix of the size of (number of sentences X number of important
words). This matrix contains the scores assigned to the terms.
This matrix is used as the input for a K-Means clustering algorithm.
Each sentence of every cluster is then assigned a cluster number. For every cluster number we query all
the important words in the sentences belonging to that particular cluster number.
For example,
Cluster[0] contains ['date',
'breakup',
'hookups',
'creepy',
'religious',
'career',
'study',
'father',
'bullshit',
'aemergency',
'dial',
'wora','brian',
'knew',
'brother',
'press',
'result',
'step',
'preferences',
'treatment',
'fault',
'mother',
'kyle',
'film',
'stopped',
'bites',
'sister',
'bitch',
'grandad',

We see this cluster contains terms like father, sister, granddad, religious, and breakup. All of these
words point to inter personal relationships, which can be a cause for stress and mental health issues.

For cluster[2] we have,
'headaches',
'whoaTMs',
'iaTMve',
'heaTMs',
'iaTMm',
'responsible',
'kinda',
'proceed',
'freak',
'horribly',
'italian',
'alcohol',
'evidence',
'ridiculous',
'theyaTMre',
'expand',
'didnaTMt',
'knew',
'putting',
'surgery',
'drained',
'eating',
'stomach',
'starting',
'sweet',

We can see words like surgery, headache, alcohol, stomach. These keywords point to medical issues
which could lead to stress or mental health issuesFor cluster 3 we have,

feel',
'finds',
'ways',
'navy',
'slow',
'chaotic',
'danger',
'locker',
'edge',
'deserve',
'daddy',
'sick',
'jokingly',
'prozac',
'leave',
'horrible',
'relax',
'overweight',
'heaTMs',
'broke',
'spaced',
'owners',
'clone',
'dont',
'funds',
'vegas',
'cured',
'experienced',
'envy',
'attack',
'panic',

Words include feel,danger,edge,envy,panic,chaotic. These words point to intense feelings of panic and
anxiety, which is a cause for stress and mental issues.
Similar readings can be made for other clusters.