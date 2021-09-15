# import nltk
# nltk.download('punkt')
# from sklearn.feature_extraction.text import CountVectorizer
# tokenizer = CountVectorizer().build_tokenizer()
# print(tokenizer("Here’s example text, isn’t it?"))

# from nltk.tokenize import WhitespaceTokenizer
# from nltk.tokenize import word_tokenize
# print(WhitespaceTokenizer().tokenize("Here’s example text, isn’t it?"))
# print(word_tokenize("Here’s example text, isn’t it?"))
# print(tokenizer("likes liking liked"))
# print(WhitespaceTokenizer().tokenize("likes liking liked"))
# print(word_tokenize("likes liking liked"))

# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# tokens = word_tokenize("Here’s example text, isn’t it?")
# stems = [stemmer.stem(token) for token in tokens]
# print(stems)
# tokens = word_tokenize("likes liking liked")
# stems = [stemmer.stem(token) for token in tokens]
# print(stems)

# import spacy
# import json_lines

# X = []; y = []; z = []
# with open ( 'final_steam_reviews.jl' , 'rb' ) as f:
#     for item in json_lines.reader( f ):
#         X.append( item['text'] )
#         y.append( item['voted_up'] )
#         z.append( item ['early_access'] )

# X_tokens = []
# for line in X:

from sklearn.metrics import accuracy_score, recall_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))