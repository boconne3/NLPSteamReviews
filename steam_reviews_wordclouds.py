import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

df3_col_names = ['text', 'upvote', 'early_access']
df3 = pd.read_csv('en_ascii_reviews.csv', comment='#', names=df3_col_names)

stopwords = set(STOPWORDS)
textt = " ".join(review for review in df3.text)
pos = ""
neg = ""
for index, row in df3.iterrows():
    if(row['upvote'] == True):
        pos += row['text'] + " "
    else:
        neg += row['text'] + " "

wordcloud = WordCloud(stopwords=stopwords).generate(textt)
pos_wordcloud = WordCloud(stopwords=stopwords).generate(pos)
neg_wordcloud = WordCloud(stopwords=stopwords).generate(neg)

plt.figure(1)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud1.png')

plt.figure(2)
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('pos_wordcloud.png')

plt.figure(3)
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('neg_wordcloud.png')
plt.show()

# <\CREATE WORD CLOUDS>