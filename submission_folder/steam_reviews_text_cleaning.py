import json_lines
from langdetect import detect, detect_langs
import pandas as pd
import numpy as np
from google_trans_new import google_translator   

def find_english(X, y, z):
    count = 0
    no_lang_count = 0
    X_en = []
    for i in range(len(X)):
        try:
            if(detect_langs(X[i])[0].lang == "en"):
                count += 1
                X_en.append([X[i], y[i], z[i]])
        except:
            no_lang_count += 1
    return X_en, count, no_lang_count

def find_non_empty(X, y, z):
    empty_count = 0
    X_non_empty = []
    for i in range(len(X)):
        if X[i] == "":
            empty_count += 1
        else:
            X_non_empty.append([X[i], y[i], z[i]])
    return X_non_empty, len(X) - empty_count, empty_count

# <OPEN DATASET>
X = []; y = []; z = []
with open ( 'final_steam_reviews.jl' , 'rb' ) as f:
    for item in json_lines.reader( f ):
        X.append( item['text'] )
        y.append( item['voted_up'] )
        z.append( item ['early_access'] )
# <\OPEN DATASET>

# <SIMPLE LANGUAGE DETECTION>
no_lang_count = 0
langs = []
lang_freq = []
for i in range(len(X)):
    try:
        lang = detect_langs(X[i])[0].lang
        if(lang in langs):
            lang_freq[langs.index(lang)] += 1
        else:
            langs.append(lang)
            lang_freq.append(1)
    except:
        no_lang_count += 1

for i in range(len(langs)):
    print(f"{langs[i]} - {lang_freq[i]}")

print(f"No language detected - {no_lang_count}")

# <\SIMPLE LANGUAGE DETECTION>

# <TRANSLATE ALL REVIEWS TO ENGLISH AND SAVE>

X_non_empty, non_empty_count, empty_count = find_non_empty(X, y, z)
print(f"Non-empty count: {non_empty_count}")
print(f"Empty count: {empty_count}")

translator = google_translator() 
X_en = []

translated_texts = []
y_trans = []
z_trans = []

start_index = 0     # variable to allow for translation to be run in chunks
df1 = pd.read_csv('translated_reviews.csv', comment='#', columns=['index','reviews', 'upvoted', 'early_access'])
for i in range(len(X_non_empty) - start_index):
    curr_index = i + start_index
    translate_text = translator.translate(X_non_empty[curr_index][0],lang_tgt='en') 
    translated_texts.append(translate_text)
    y_trans.append(y[curr_index])
    z_trans.append(z[curr_index])
    data = {'trans_reviews': translated_texts, 'upvoted': y_trans, 'early_access': z_trans}
    df = pd.DataFrame(data)
    df.to_csv('translated_reviews_all.csv', index=False, header=False)
    print(curr_index)

# <\TRANSLATE ALL REVIEWS TO ENGLISH AND SAVE>

# <FIND AND SAVE ALL NON-EMPTY ENGLISH REVIEWS>

X_non_empty, non_empty_count, empty_count = find_non_empty(X)
print(f"Non-empty count: {non_empty_count}")
print(f"Empty count: {empty_count}")

X_en, count, no_lang_count = find_english(X_non_empty)
print(f"English count: {count}")
print(f"No language count: {no_lang_count}")

df = pd.DataFrame(np.array(X_en), columns=['en_reviews', 'upvoted', 'early_access'])
df.to_csv('english_reviews.csv', index=False, header=False)

# <\FIND AND SAVE ALL NON-EMPTY ENGLISH REVIEWS>

# <REMOVE ALL NON-ASCII CHARS FROM ENGLISH REVIEWS AND SAVE>

df_col_names = ['review', 'upvote', 'early_access']
df = pd.read_csv('translated_reviews_all.csv', comment='#', names=df_col_names)

for index, row in df.iterrows():
    row['review'] = row['review'].encode('ascii',errors='ignore').decode('ascii')

df.to_csv('translated_reviews_all_ascii.csv', index=False, header=False)

# <\REMOVE ALL NON-ASCII CHARS FROM ENGLISH REVIEWS AND SAVE>