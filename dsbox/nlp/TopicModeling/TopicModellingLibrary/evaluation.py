from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import calinski_harabaz_score, davies_bouldin_score


def calc_WCSS(word_array, assigned_clusters, n_components):
    '''
    Calculates the sum of squared distances of samples to their closest cluster center

    Parameters
    ----------
    word_array: Numpy array of the n-dimensional sample
    assigned_clusters: Cluster number (starting from 0) assigned to each sample
    n_components: Number of clusters

    Returns
    ----------
    WCSS_value: Sum of squared distances of samples to their closest cluster center
    '''

    centroids = np.array([np.mean(word_array[assigned_clusters == cluster], axis=0) for cluster in range(n_components)])

    WCSS_value = 0
    for index,sample in enumerate(word_array):
        distance_from_closest_centroid = np.linalg.norm(sample - centroids[assigned_clusters[index]])
        WCSS_value += distance_from_closest_centroid**2

    return WCSS_value


def dice_coefficient(a, b):
    """dice coefficient 2nt/(na + nb)."""
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))


def get_relevant_terms(corpus, topics, method = 'TextRank', param = 1):
    '''
    Parameters
    ----------
    corpus: List of documents
    topics: Topics assigned to each document in the corpus
    param: Lambda parameter. Has to be between 0 and 1.
            The closer this is to 1, the more importance is given to term frequency in the assigned topic and lesser
            importance is given to how uniquely it differentiates the current topics over other topics.

    Returns
    ----------
    relevant_words
    '''

    corpus_terms = (" ".join(corpus).split(" "))
    unique_terms = np.unique(corpus_terms)

    most_relevant_words = []
    most_relevant_doc = []
    topic_names = []

    if method == 'relevance':

        for topic in np.unique(topics):

            relevance_score = []
            corpus_topic = [doc for (idx,doc) in enumerate(corpus) if topics[idx] == topic]
            original_text_topic = [doc for (idx,doc) in enumerate( list(english_comments.Comment.unique())) if topics[idx] == topic]
            corpus_topic_words =  (" ".join(corpus_topic).split(" "))

            for term in unique_terms:

                p_term_topic = corpus_topic_words.count(term)/len(corpus_topic_words)
                p_term = corpus_terms.count(term)/len(corpus_terms)
                relevance = param*np.log(p_term_topic) + (1-param)*np.log((p_term_topic+1e-10)/(p_term+1e-10))

                relevance_score.append(relevance)

            relevance_list = ([unique_terms[idx] if len(unique_terms[idx]) > 2 else '' for idx in np.argsort(relevance_score)[::-1]])
            relevance_list = list(filter(lambda x: x != '', relevance_list))

            shortlisted_terms = relevance_list[:10]

            clean_terms = []
            for term in shortlisted_terms:
                edit_distance = [dice_coefficient(term, word) for word in original_text_terms]
                clean_terms.append(original_text_terms[np.argmin(edit_distance)])

            most_relevant_words.append(np.unique(list(filter(lambda x: len(x) > 2, clean_terms))))
            topic_names.append(" ".join(np.unique(list(filter(lambda x: len(x) > 2, clean_terms)))[:3]))

            doc_score_list = []
            length_doc = []

            important_words = relevance_list[:50]

            for doc in corpus_topic:
                doc_score = 0
                for word in np.unique(doc.split(' ')):
                    if (word in important_words):
                        doc_score += relevance_score[list(unique_terms).index(word)]

                doc_score_list.append(doc_score)
                length_doc.append(len(doc))

            max_score_indices = np.where(doc_score_list == min(doc_score_list))
    #         min_length_idx = np.argmin([length_doc[idx] for idx in max_score_indices])

            most_relevant_doc.append(original_text_topic[np.argmin(doc_score_list)])

    else:
        labels = topics

        most_relevant_words = [keywords(". ".join([" ".join(doc.split()) for (idx, doc) in enumerate(corpus) if labels[idx] == topic]),
         words = 3).split('\n') for topic in np.unique(topics)]

        most_relevant_doc = [summarize(". ".join([" ".join(doc.split()) for (idx, doc) in enumerate(english_comments.Comment.unique()) if labels[idx] == topic]),
        word_count = 100).replace('\n','') if summarize(". ".join([" ".join(doc.split()) for (idx, doc) in enumerate(english_comments.Comment.unique()) if labels[idx] == topic]),
        word_count = 100).replace('\n','') != "" else (". ".join([" ".join(doc.split()) for (idx, doc) in enumerate(english_comments.Comment.unique()) if labels[idx] == topic]))[:500] for topic in np.unique(topics)]

        topic_names = [(" ".join(relevant_words for relevant_words in list_[:3])) for list_ in most_relevant_words]

        new_topics = []
        for term in topic_names:
            clean_words = []
            for word_ in term.split(" "):
                coefficient = [dice_coefficient(word_, word) for word in original_text_terms]
                clean_words.append(original_text_terms[np.argmax(coefficient)])

            new_topics.append(" ".join(clean_words))

    relevant_words = pd.DataFrame()
    relevant_words['Topic'] = np.unique(topics)
    relevant_words['RelevantWords'] = most_relevant_words
    relevant_words['Topic Title'] = new_topics
    relevant_words['Comment'] = most_relevant_doc

    return relevant_words


