import word2vec
from sklearn.manifold import TSNE
from adjustText import adjust_text
import nltk
import numpy as np
import matplotlib.pyplot as plt

MIN_COUNT = 10           # default: 5
WORDVEC_DIM = 100        # default: 100
WINDOW = 10              # default: 5
NEGATIVE_SAMPLES = 0     # default: 0
ITERATIONS = 5
MODEL = 1                # default: 1 (skip-gram model) 
LEARNING_RATE = 0.045    # default: 0.025 (learning rate)

word2vec.word2phrase('./Book5TheOrderOfThePhoenix/all.txt',
                     './wordvec/all_phrase.txt',
                     verbose = True
                    )
word2vec.word2vec('./wordvec/all_phrase.txt',
                  './wordvec/all.bin',
                    cbow = MODEL,
                    size = WORDVEC_DIM,
                    min_count = MIN_COUNT,
                    window = WINDOW,
                    negative=NEGATIVE_SAMPLES,
                    iter_=ITERATIONS,
                    alpha=LEARNING_RATE,
                    verbose = True
                 )

print('Finish Training')
model = word2vec.load('./wordvec/all.bin')
print('Load model')
plot_num = 1200
vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:plot_num]
vocabs = vocabs[:plot_num]

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]

plt.figure()
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
        x, y = reduced[i,:]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.show()
a=1
