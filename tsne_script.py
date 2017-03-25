import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import pickle as pkl
import seaborn as sns

feat_dict = pkl.load(open('feat.p', 'rb'))
features = feat_dict['f']
features = np.array(features)
shape = features.shape
features = np.reshape(features, [shape[0], shape[2]])

lab_dict = pkl.load(open('lab.p', 'rb'))
labels = lab_dict['l']
labels = np.array(sorted(labels))
print(labels.shape)
ma, mi = labels.max(), labels.min()
class_labels = []
for i in range(len(labels)):
    label = int((labels[i] - mi)/(ma-mi)*10)
    class_labels.append(label)

print(len(class_labels))
# labels = np.argmax(labels, axis=1)
# print(len(features), len(features[0]), len(features[0][0]))

# features = np.reshape(features, [len(features), len(features[0][0])])
xt = TSNE(learning_rate=100).fit_transform(features)

plt.figure(figsize=(10,10))
plt.scatter(xt[:, 0], xt[:, 1], c=class_labels)
plt.savefig("tsne_test_scratch_1.png")

#reference: this function (scatter) has been taken from https://github.com/oreillymedia/t-SNE-tutorial
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 11))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(11):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

labels = np.array(labels)
class_labels = np.array(class_labels)
print(class_labels.max(), class_labels.min())
scatter(xt, class_labels)

plt.savefig("tsne_test_exotic_scratch_500.png")
