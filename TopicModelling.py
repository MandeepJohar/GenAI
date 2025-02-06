import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#Download NLTK stopwords (run only once)
nltk.download('punkt')
nltk.download('stopwords')

#Sample documents
documents = [
     "Artificial intelligence is transforming the technology industry.",
    "Machine learning and AI are shaping the future of automation.",
    "Deep learning algorithms are a subset of machine learning.",
    "Quantum computing will revolutionize industries like AI.",
    "Healthcare is benefiting from AI and machine learning advances.",
]

#Preprocess the documents
def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(doc.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

processed_docs = [preprocess(doc) for doc in documents]

# Create a dictionary and document-term matrix
dictionary = corpora.Dictionary(processed_docs)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_docs]


# Train the LDA model(specifying 4 topics in num_topics)
lda_model = LdaModel(doc_term_matrix, num_topics=4, id2word=dictionary, passes=15)

#Print the topics with associated words , num_words=2 mentioning the topic name in 2 words
print("Topics discovered by LDA:")
topics = lda_model.print_topics(num_words=2)
for topic in topics:
  print(topic)

#Document similarity between doc1_bow and doc2_bow
doc1_bow = dictionary.doc2bow(preprocess("AI and machine learning are advancing rapidly"))
doc2_bow = dictionary.doc2bow(preprocess("Machine learning and AI are shaping the future of automation."))

similarity = gensim.matutils.cossim(doc1_bow, doc2_bow)
print("\nDocument Similarity (cosine):", similarity)
