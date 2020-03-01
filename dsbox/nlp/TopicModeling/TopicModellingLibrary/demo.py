import pandas as pd
from learn import learner

corpus = pd.read_excel('Data_Train.xlsx').STORY

model = learner(corpus.values)
model.preprocess()
model.vectorize()
model.learn('LDA', )
print (type(corpus))