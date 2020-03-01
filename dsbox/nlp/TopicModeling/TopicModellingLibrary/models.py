# class attributes: n_topics, random_state
# class methods: fit, transform

class TMAlgorithm(object):

    def __init__(self, n_topics=5, random_state=42):
        self.n_topics = n_topics
        self.random_state = random_state

    def fit(self, vectorized):
        '''
        Fits feature matrix with n_topics
        returns self
        '''
        self.vectorized = vectorized
        pass

    def transform(self, vectorized):
        '''
        Determines topics from the feature matrix
        returns self.clusters
        '''
        pass


class LDA(TMAlgorithm):

    def __init__(self):
        super().__init__()
        self.model = LatentDirichletAllocation(n_components = self.n_topics, random_state = self.random_state)

    def fit(self, vectorized):
        self.model.fit(vectorized)

    def transform(self, vectorized):
        return self.model.transform(vectorized).argmax(axis=1)