
train_file = "motion.wsd"

lines = None
with open(train_file, "r") as train:
    lines = train.readlines()


# In[2]:


instance_set = []

instance = ""
for each_line in lines:
    stripped_line = each_line.rstrip("\n")
    instance += stripped_line
    if stripped_line == "</instance>":
        instance_set.append(instance)
        instance = ""
        
num_of_instance = len(instance_set)
print(f"Found {num_of_instance} instance(s)")


# In[3]:


import re
import numpy as np

vocabulary = []

for i in range(len(instance_set)):
    sentence = instance_set[i].split("context")[1].strip("[></ ]").lower()
    sentence = re.sub(" <head>", "", sentence)
    sentence = re.sub("</head> ", "", sentence)
    sentence = re.sub("[().?:;,+=!$&*-]", "", sentence)
    sentence = re.sub("[0-9]", "", sentence)
    
    sense = "physical" if "motion%physical" in instance_set[i] else "legal"
    for word in sentence.split(" "):
        if word not in ['']:
            if len(word) >= 2:
                word = word.strip()
                vocabulary.append([word, sense])
    
vocabulary = np.asarray(vocabulary)


# In[4]:


def get_validation_folds(vocabulary: np.ndarray, k = 5):
    dataset_folds = []
    jump_by = vocabulary.shape[0] // k
    start = 0
    end = jump_by
    for i in range(k):
        fold = vocabulary[start:end, :]
        dataset_folds.append(fold)
        start = end
        end = end + jump_by

    return np.asarray(dataset_folds)


# In[5]:


def accuracy(predicted, original):
    if len(predicted) != len(original):
        raise Exception("Prediction set and original set must have number of samples")
    
    hits = 0
    n_samples = len(original)
    for x, y in zip(predicted, original):
        if x == y:
            hits += 1
    
    return round(hits / n_samples, 3)


def sense_probability(dataset):
    sense, frequency = np.unique(dataset[:, 1], return_counts = True)
    sense = list(sense)
    frequency = list(frequency)
    probab = []
    nrows = dataset.shape[0]
    for i in range(len(sense)):
        p = round(frequency[i] / nrows, 3)
        probab.append(p)
        
    return np.array([sense, probab])


# In[6]:


def split(dataset, test_sample = 0):
    train = []
    test = []
    for k in dataset[test_sample]:
        test.append(list(k))
    
    for i in range(len(dataset)):
        if i == test_sample:
            continue
        for k in dataset[i]:
            train.append(list(k))

    return np.asarray(train), np.asarray(test)


# In[7]:


# training and testing
dataset_folds = get_validation_folds(vocabulary, k = 5)

k = 5
accuracies = []
for i in range(k):
    train, test = split(dataset_folds, test_sample = i)
    probab = sense_probability(train)
    
    # testing
    unique_words = np.unique(test[:, 0])
    word_dict = {}
    for word in unique_words:
        word_dict[word] = [k for k, w in enumerate(test) if w[0] == word]

    feature_set = []
    for word in unique_words:
        sense, count = np.unique(test[word_dict[word], 1], return_counts = True)
        if len(count) != 2:
            feature = []
            if sense[0] == 'physical':
                feature = [count[0], 0, np.argmax([count[0], 0])]
            else:
                feature = [0, count[0], np.argmax([0, count[0]])]
            feature_set.append(feature)
        else:
            feature_set.append([count[0], count[1], np.argmax(count)])
    
    feature_set = np.asarray(feature_set)
    scores = feature_set[:, 0:2] * probab[1].astype(float)
    
    res = []
    for j in range(len(scores)):
        result = [unique_words[j], np.argmax(scores[j])]
        res.append(result)
    
    res = np.asarray(res)
    acc = accuracy(res[:, 1].astype(int), feature_set[:, 2].astype(int)) * 100
    accuracies.append(acc)
    print("Training accuracy: fold {0} - {1}".format(i + 1, acc))
    
print("Average accuracy for Motion Dataset: ", np.mean(accuracies))


output_file = "Motion.wsd.out"
sense_names = ["motion%physical", "motion%legal"]
with open(output_file, "w") as output:
    for word, sense in res:
        output.write(f"{word}: {sense_names[int(sense)]}\n")

# Print a message to confirm that the results have been saved
print(f"Results saved to '{output_file}'")




