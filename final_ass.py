from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def softmax(y):
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = -1
    return y

def confusion_matrix(y_pred, y):
    tp = tn = fp = fn = 0
    for idx, val in enumerate(y_pred):
        if val == -1 and y[idx] == -1:
            tn += 1
        elif val == 1 and y[idx] == 1:
            tp += 1
        elif val == -1 and y[idx] == 1:
            fn += 1
        elif val == 1 and y[idx] == -1:
            fp += 1
    return tp, tn, fp, fn

def accuracy_calc(conf_tuple):
    tp = conf_tuple[0]
    tn = conf_tuple[1]
    fp = conf_tuple[2]
    fn = conf_tuple[3]
    return (tp+tn)/(tp+tn+fp+fn)

def calc_shuffle_order(length):
    shuffle_order = np.arange(length)
    random.shuffle(shuffle_order)
    return shuffle_order

def shuffle2(X, y, shuffle_order):
    X_shuff = []
    y_shuff = []
    for i in shuffle_order:
        X_shuff.append(X[i])
        y_shuff.append(y[i])
    return np.array(X_shuff), np.array(y_shuff)

def unshuffle1(X_shuff, shuffle_order):
    X = []
    for i in range(len(shuffle_order)):
        X.append(X_shuff[np.where(shuffle_order == i)[0][0]])
    return np.array(X)


vect = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=1500, stop_words='english')

df_col_names = ['text', 'upvote', 'early_access']
df = pd.read_csv('en_ascii_reviews.csv', comment='#', names=df_col_names)
# df = pd.read_csv('translated_reviews_all_ascii.csv', comment='#', names=df_col_names)

transform = vect.fit_transform(df.text)
X_orig = np.array(transform.toarray())

y = []
for index, row in df.iterrows():
    if(row['upvote'] == True):
    # if(row['early_access'] == True):
        y.append(1)
    else:
        y.append(-1)
y_orig = np.array(y)

print(f"All Features - {vect.get_feature_names()}")
print(f"Num input features - {len(X_orig[0])}\nNum rows - {len(X_orig)}")
print(f"Num output rows - {len(y_orig)}")

# <BEST-FIT MODEL COMPARISONS>

lr_model = LogisticRegression(penalty='l2',C=1, max_iter=1000)
knn_model = KNeighborsClassifier(n_neighbors=4)
rf_model = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, shuffle=True)

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

ns_probs = [0 for _ in range(len(y_test))]
lr_probs = lr_model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
knn_probs = knn_model.predict_proba(X_test)
knn_probs = knn_probs[:, 1]
rf_probs = rf_model.predict_proba(X_test)
rf_probs = rf_probs[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('4-NN: ROC AUC=%.3f' % (knn_auc))
print('Logistic Reg ROC AUC=%.3f' % (lr_auc))
print('Random Forest ROC AUC=%.3f' % (rf_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

train_acc_lr = accuracy_score(lr_model.predict(X_train), y_train)
test_acc_lr = accuracy_score(lr_model.predict(X_test), y_test)
train_acc_knn = accuracy_score(knn_model.predict(X_train), y_train)
test_acc_knn = accuracy_score(knn_model.predict(X_test), y_test)
train_acc_rf = accuracy_score(rf_model.predict(X_train), y_train)
test_acc_rf = accuracy_score(rf_model.predict(X_test), y_test)

print(f"Log Reg: training accuracy - {train_acc_lr}")
print(f"Log Reg: validation accuracy - {test_acc_lr}")
print(f"4NN: training accuracy - {train_acc_knn}")
print(f"4NN: validation accuracy - {test_acc_knn}")
print(f"Random forest: training accuracy - {train_acc_rf}")
print(f"Random forest: validation accuracy - {test_acc_rf}")

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Log. Reg. C=1')
plt.plot(knn_fpr, knn_tpr, marker='x', label='4NN')
plt.plot(rf_fpr, rf_tpr, marker='o', label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC curve comparison')
plt.show()

# <\BEST-FIT MODEL COMPARISONS>

# <LOGISTIC REGRESSION CROSS-VAL>

# costs = [0.1, 0.25, 0.5, 0.75, 1, 2, 5, 7.5, 10]
costs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

logit_shuff_train_metrics = []
logit_shuff_test_metrics = []
logit_shuff_train_variances = []
logit_shuff_test_variances = []
logit_unshuff_train_metrics = []
logit_unshuff_test_metrics = []
logit_unshuff_train_variances = []
logit_unshuff_test_variances = []

for cost in costs:
    for shuffle in [True]:
        if(not shuffle):
            print("UNSHUFFLED")
            X = X_orig
            y = y_orig
        else:
            X, y = shuffle2(X_orig, y_orig, calc_shuffle_order(len(X_orig)))
            print("SHUFFLED")

        kf = KFold(n_splits=5)
        temp_train_metrics = []
        temp_test_metrics = []
        for train, test in kf.split(X):
            logit_model = LogisticRegression(penalty='l2',C=cost, max_iter=1000)
            logit_model.fit(X[train],y[train])

            y_hat_train = logit_model.predict(X[train])
            y_hat_test = logit_model.predict(X[test])

            y_hat_train_logit_acc = accuracy_score(y_hat_train, y[train])
            y_hat_test_logit_acc = accuracy_score(y_hat_test, y[test])

            y_hat_train_logit_rec = recall_score(y_hat_train, y[train])
            y_hat_test_logit_rec = recall_score(y_hat_test, y[test])

            temp_train_metrics.append([y_hat_train_logit_acc, y_hat_train_logit_rec])
            temp_test_metrics.append([y_hat_test_logit_acc, y_hat_test_logit_rec])


        logit_train_metrics = [np.mean([row[0] for row in temp_train_metrics]), np.mean([row[1] for row in temp_train_metrics])]
        logit_test_metrics = [np.mean([row[0] for row in temp_test_metrics]), np.mean([row[1] for row in temp_test_metrics])]
        logit_train_variance = [np.var([row[0] for row in temp_train_metrics]), np.var([row[1] for row in temp_train_metrics])]
        logit_test_variance = [np.var([row[0] for row in temp_test_metrics]), np.var([row[1] for row in temp_test_metrics])]

        # print(f"Some model coefs - {model.coef_[0]}")
        print("LOGISTIC REG")
        print(f"COST - {cost}")
        print(f"Training accuracy - {logit_train_metrics[0]} +/- {logit_train_variance[0]}")
        print(f"Validation accuracy - {logit_test_metrics[0]} +/- {logit_test_variance[0]}")
        print(f"Training recall - {logit_train_metrics[1]} +/- {logit_train_variance[1]}")
        print(f"Validation recall - {logit_test_metrics[1]} +/- {logit_test_variance[1]}\n")

        if(not shuffle):
            logit_unshuff_train_metrics.append(logit_train_metrics)
            logit_unshuff_test_metrics.append(logit_test_metrics)
            logit_unshuff_train_variances.append(logit_train_variance)
            logit_unshuff_test_variances.append(logit_test_variance)
        else:
            logit_shuff_train_metrics.append(logit_train_metrics)
            logit_shuff_test_metrics.append(logit_test_metrics)
            logit_shuff_train_variances.append(logit_train_variance)
            logit_shuff_test_variances.append(logit_test_variance)

plt.figure(1)
# plt.errorbar(costs, [row[0] for row in logit_unshuff_train_metrics], yerr=[row[0] for row in logit_unshuff_train_variances], color='r', ecolor='k', label="Unshuffled - train")
plt.errorbar(costs, [row[0] for row in logit_shuff_train_metrics], yerr=[row[0] for row in logit_shuff_train_variances], color='b', ecolor='k', label="Training")
# plt.errorbar(costs, [row[0] for row in logit_unshuff_test_metrics], yerr=[row[0] for row in logit_unshuff_test_variances], color='orange', ecolor='k', label="Unshuffled - val")
plt.errorbar(costs, [row[0] for row in logit_shuff_test_metrics], yerr=[row[0] for row in logit_shuff_test_variances], color='c', ecolor='k', label="Validation")
plt.legend()
plt.xlabel("Cost")
plt.ylabel("Accuracy")
plt.title("Logistic regression - Sentiment analysis accuracy")
plt.xscale("log")
plt.savefig('logistic_acc_cv_early_acc_all.png')


plt.figure(2)
# plt.errorbar(costs, [row[1] for row in logit_unshuff_train_metrics], yerr=[row[1] for row in logit_unshuff_train_variances], color='r', ecolor='k', label="Unshuffled - train")
plt.errorbar(costs, [row[1] for row in logit_shuff_train_metrics], yerr=[row[1] for row in logit_shuff_train_variances], color='b', ecolor='k', label="Training")
# plt.errorbar(costs, [row[1] for row in logit_unshuff_test_metrics], yerr=[row[1] for row in logit_unshuff_test_variances], color='orange', ecolor='k', label="Unshuffled - val")
plt.errorbar(costs, [row[1] for row in logit_shuff_test_metrics], yerr=[row[1] for row in logit_shuff_test_variances], color='c', ecolor='k', label="Validation")
plt.legend()
plt.xlabel("Cost")
plt.ylabel("Recall")
plt.title("Logistic regression - Sentiment analysis recall")
plt.xscale("log")
plt.savefig('logistic_rec_cv_early_acc_all.png')
plt.show()

# <\LOGISTIC REGRESSION CROSS-VAL>

# <KNN CROSS-VAL>

knn_unshuff_train_metrics = []
knn_unshuff_test_metrics = []
knn_unshuff_train_variances = []
knn_unshuff_test_variances = []
knn_shuff_train_metrics = []
knn_shuff_test_metrics = []
knn_shuff_train_variances = []
knn_shuff_test_variances = []

neighbours = [1,2,3,4,5,6,7,8,9]
for neighbour in neighbours:
    for shuffle in [True]:
        if(not shuffle):
            print("UNSHUFFLED")
            X = X_orig
            y = y_orig
        else:
            X, y = shuffle2(X_orig, y_orig, calc_shuffle_order(len(X_orig)))
            print("SHUFFLED")

        kf = KFold(n_splits=5)
        temp_train_metrics = []
        temp_test_metrics = []
        for train, test in kf.split(X):
            knn_model = KNeighborsClassifier(n_neighbors=neighbour)
            knn_model.fit(X[train],y[train])

            y_hat_train = knn_model.predict(X[train])
            y_hat_test = knn_model.predict(X[test])

            y_hat_train_logit_acc = accuracy_score(y_hat_train, y[train])
            y_hat_test_logit_acc = accuracy_score(y_hat_test, y[test])

            y_hat_train_logit_rec = recall_score(y_hat_train, y[train])
            y_hat_test_logit_rec = recall_score(y_hat_test, y[test])

            temp_train_metrics.append([y_hat_train_logit_acc, y_hat_train_logit_rec])
            temp_test_metrics.append([y_hat_test_logit_acc, y_hat_test_logit_rec])


        knn_train_metrics = [np.mean([row[0] for row in temp_train_metrics]), np.mean([row[1] for row in temp_train_metrics])]
        knn_test_metrics = [np.mean([row[0] for row in temp_test_metrics]), np.mean([row[1] for row in temp_test_metrics])]
        knn_train_variance = [np.var([row[0] for row in temp_train_metrics]), np.var([row[1] for row in temp_train_metrics])]
        knn_test_variance = [np.var([row[0] for row in temp_test_metrics]), np.var([row[1] for row in temp_test_metrics])]

        print("kNN")
        print(f"Neighbours - {neighbour}")
        print(f"Training accuracy - {knn_train_metrics[0]} +/- {knn_train_variance[0]}")
        print(f"Validation accuracy - {knn_test_metrics[0]} +/- {knn_test_variance[0]}")
        print(f"Training recall - {knn_train_metrics[1]} +/- {knn_train_variance[1]}")
        print(f"Validation recall - {knn_test_metrics[1]} +/- {knn_test_variance[1]}\n")

        if(not shuffle):
            knn_unshuff_train_metrics.append(knn_train_metrics)
            knn_unshuff_test_metrics.append(knn_test_metrics)
            knn_unshuff_train_variances.append(knn_train_variance)
            knn_unshuff_test_variances.append(knn_test_variance)
        else:
            knn_shuff_train_metrics.append(knn_train_metrics)
            knn_shuff_test_metrics.append(knn_test_metrics)
            knn_shuff_train_variances.append(knn_train_variance)
            knn_shuff_test_variances.append(knn_test_variance)

plt.figure(1)
# plt.errorbar(neighbours, [row[0] for row in knn_unshuff_train_metrics], yerr=[row[0] for row in knn_unshuff_train_variances], color='r', ecolor='k', label="Unshuffled - train")
plt.errorbar(neighbours, [row[0] for row in knn_shuff_train_metrics], yerr=[row[0] for row in knn_shuff_train_variances], color='b', ecolor='k', label="Training")
# plt.errorbar(neighbours, [row[0] for row in knn_unshuff_test_metrics], yerr=[row[0] for row in knn_unshuff_test_variances], color='orange', ecolor='k', label="Unshuffled - val")
plt.errorbar(neighbours, [row[0] for row in knn_shuff_test_metrics], yerr=[row[0] for row in knn_shuff_test_variances], color='c', ecolor='k', label="Validation")
plt.legend()
plt.xlabel("Neighbours")
plt.ylabel("Accuracy")
plt.title("kNN - Sentiment analysis accuracy")
plt.savefig('knn_acc_cv_early_acc_all.png')


plt.figure(2)
# plt.errorbar(neighbours, [row[1] for row in knn_unshuff_train_metrics], yerr=[row[1] for row in knn_unshuff_train_variances], color='r', ecolor='k', label="Unshuffled - train")
plt.errorbar(neighbours, [row[1] for row in knn_shuff_train_metrics], yerr=[row[1] for row in knn_shuff_train_variances], color='b', ecolor='k', label="Training")
# plt.errorbar(neighbours, [row[1] for row in knn_unshuff_test_metrics], yerr=[row[1] for row in knn_unshuff_test_variances], color='orange', ecolor='k', label="Unshuffled - val")
plt.errorbar(neighbours, [row[1] for row in knn_shuff_test_metrics], yerr=[row[1] for row in knn_shuff_test_variances], color='c', ecolor='k', label="Validation")
plt.legend()
plt.xlabel("Neighbours")
plt.ylabel("Recall")
plt.title("kNN - Sentiment analysis recall")
plt.savefig('knn_rec_cv_early_acc_all.png')
plt.show()

# <\KNN CROSS-VAL>

# <RANDOM FOREST K-FOLD ANALYSIS>

X, y = shuffle2(X_orig, y_orig, calc_shuffle_order(len(X_orig)))
print("SHUFFLED")

kf = KFold(n_splits=5)
temp_train_metrics = []
temp_test_metrics = []
for train, test in kf.split(X):
    rf_model = RandomForestClassifier(max_depth=None, random_state=0)
    rf_model.fit(X[train],y[train])

    y_hat_train = rf_model.predict(X[train])
    y_hat_test = rf_model.predict(X[test])

    y_hat_train_rf_acc = accuracy_score(y_hat_train, y[train])
    y_hat_test_rf_acc = accuracy_score(y_hat_test, y[test])

    y_hat_train_rf_rec = recall_score(y_hat_train, y[train])
    y_hat_test_rf_rec = recall_score(y_hat_test, y[test])

    temp_train_metrics.append([y_hat_train_rf_acc, y_hat_train_rf_rec])
    temp_test_metrics.append([y_hat_test_rf_acc, y_hat_test_rf_rec])


rf_train_metrics = [np.mean([row[0] for row in temp_train_metrics]), np.mean([row[1] for row in temp_train_metrics])]
rf_test_metrics = [np.mean([row[0] for row in temp_test_metrics]), np.mean([row[1] for row in temp_test_metrics])]
rf_train_variance = [np.var([row[0] for row in temp_train_metrics]), np.var([row[1] for row in temp_train_metrics])]
rf_test_variance = [np.var([row[0] for row in temp_test_metrics]), np.var([row[1] for row in temp_test_metrics])]

print("RANDOM FOREST")
print(f"Training accuracy - {rf_train_metrics[0]} +/- {rf_train_variance[0]}")
print(f"Validation accuracy - {rf_test_metrics[0]} +/- {rf_test_variance[0]}")
print(f"Training recall - {rf_train_metrics[1]} +/- {rf_train_variance[1]}")
print(f"Validation recall - {rf_test_metrics[1]} +/- {rf_test_variance[1]}\n")

# <\RANDOM FOREST K-FOLD ANALYSIS>

# <BASELINE K-FOLD ANALYSIS>

X, y = shuffle2(X_orig, y_orig, calc_shuffle_order(len(X_orig)))
print("UNSHUFFLED")
kf = KFold(n_splits=5)
temp_train_metrics = []
temp_test_metrics = []
for train, test in kf.split(y):
    train_common_class = statistics.mode(y[train])

    y_hat_train = np.ones(len(y[train]), dtype=np.int32) * train_common_class
    y_hat_test = np.ones(len(y[test]), dtype=np.int32) * train_common_class

    y_hat_train_baseline_acc = accuracy_score(y[train], y_hat_train)
    y_hat_test_baseline_acc = accuracy_score(y[test], y_hat_test)

    y_hat_train_baseline_rec = recall_score(y[train], y_hat_train)
    y_hat_test_baseline_rec = recall_score(y[test], y_hat_test)

    temp_train_metrics.append([y_hat_train_baseline_acc, y_hat_train_baseline_rec])
    temp_test_metrics.append([y_hat_test_baseline_acc, y_hat_test_baseline_rec])

baseline_train_metrics = [np.mean([row[0] for row in temp_train_metrics]), np.mean([row[1] for row in temp_train_metrics])]
baseline_test_metrics = [np.mean([row[0] for row in temp_test_metrics]), np.mean([row[1] for row in temp_test_metrics])]
baseline_train_variance = [np.var([row[0] for row in temp_train_metrics]), np.var([row[1] for row in temp_train_metrics])]
baseline_test_variance = [np.var([row[0] for row in temp_test_metrics]), np.var([row[1] for row in temp_test_metrics])]
print("BASELINE")
print(f"Training accuracy - {baseline_train_metrics[0]} +/- {baseline_train_variance[0]}")
print(f"Validation accuracy - {baseline_test_metrics[0]} +/- {baseline_test_variance[0]}")
print(f"Training recall - {baseline_train_metrics[1]} +/- {baseline_train_variance[1]}")
print(f"Validation recall - {baseline_test_metrics[1]} +/- {baseline_test_variance[1]}\n")

# <\BASELINE K-FOLD ANALYSIS>
