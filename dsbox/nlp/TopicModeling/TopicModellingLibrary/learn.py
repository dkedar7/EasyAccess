from preprocess import text_preprocessing
import evaluation
import vectorizer
import joblib
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def choose_optimal_topics(combined_metrics, min_members, max_members, corpus):
    '''
    Function to choose the optimal number of topics using the following rules:
      Rule 1.There is a lower limit on the number of datapoints in a cluster.
               Default value is 2% of the lenth of the corpus.
      Rule 2.There is an upper limit on the number of datapoints in a cluster.
               Default value is 50% of the lenth of the corpus.
      Rule 3. The chosen number of topics is lesser than 7% of the size of corpus.
    '''
    optimal_value = -100
    chosen_topics = -1
    for idx, value in enumerate(combined_metrics):
        if (value > optimal_value) \
        and (min_members[idx] > 0.02*len(corpus)) \
        and (max_members[idx] < 0.5*len(corpus)) \
        and (list(range(2,50))[idx] < 0.07*len(corpus)):
            optimal_value = value
            chosen_topics = list(range(2,50))[idx]
    return chosen_topics


class learner(object):

	def __init__(self, corpus):
		self.corpus = corpus
		self.Train = True

	def preprocess(self, di_regex = joblib.load('di_regex.pkl'), additional_stop_words = False, 
	fl_lemmatize = False, fl_stemmer = False):

		stop_words = stopwords.words('english')

		if additional_stop_words:
			stop_words.extend(additional_stop_words)
			
		self.corpus = text_preprocessing(self.corpus, di_regex, stop_words, fl_lemmatize, fl_stemmer)

		return self

	def train(self): 
		self.Train = True

	def eval(self):
		self.Train = False

	def vectorize(self):

		# This is defined in the vectorizer
		vector = vectorizer.TfidfVectorizer(min_df = 0.1,
                                 max_df = 0.5,
                                 stop_words = 'english')

		self.vectorized = vector.fit_transform(self.corpus)

		return self

	# Which algorithm is to be used
	# def algorithm(self, algorithm): self.algorithm = algorithm

	def learn(self, algorithm, n_topics = 5, random_state = 42):
		'''
		Do topic modeling.
		Returns tuple with text and assgned topics 
		'''
		self.algorithm = algorithm
		self.n_topics = n_topics
		# model = self.algorithm(n_components = n_topics, random_state = random_state)
		# model.fit(self.vectorized)

		if self.Train:
			algorithm.fit(self.vectorized)

		self.clusters = algorithm.transform(self.vectorized)

		return self.corpus, self.clusters

	def evaluate(self):
		'''
		Calculate all metrics
		'''
		WCSS = evaluation.calc_WCSS(self.vectorized.toarray(), self.clusters, comps)
		silhoutte = evaluation.silhouette_score(self.vectorized.toarray(), self.clusters, metric='cosine')
		ch_score = evaluation.calinski_harabaz_score(self.vectorized.toarray(), self.clusters)
		db_score = evaluation.davies_bouldin_score(self.vectorized.toarray(), self.clusters)
		min_members = np.min(np.unique(self.clusters, return_counts=True)[1])
		max_members = np.max(np.unique(self.clusters, return_counts=True)[1])

		self.metrics = {}
		self.metrics['Within-Cluster-Sum-of-Squares (WCSS)'] = WCSS
		self.metrics['Mean Silhoutte Coefficient'] = silhoutte
		self.metrics['Calinski and Harabasz score'] = ch_score
		self.metrics['Davies-Bouldin score'] = db_score
		self.metrics['Size of the Smallest Cluster'] = min_members
		self.metrics['Size of the Largest Cluster'] = max_members

		return self.metrics

	def experiment(self):
		'''
		Evaluate using all TM algorithms, determine the optimal number of topics
		using 'choose_optimal_topics'
		'''
		pass


	




	
