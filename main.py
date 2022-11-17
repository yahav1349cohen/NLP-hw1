import pickle
from preprocessing import preprocess_train, read_test
from optimization import get_optimal_vector
from inference import tag_all_test, memm_viterbi_with_beam_search
from tqdm import tqdm
import functools
from sklearn import metrics
import pandas as pd


def learn():
    threshold = 1
    lam = 1

    train_path = "train1.wtag"
    test_path = "comp1.words"

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    print("accuracy of model with capital letter feature and number detection feature")
    tag_all_test_with_accuracy_and_confusion_matrix('test1.wtag', pre_trained_weights, feature2id)
    # tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


def tag_all_test_with_accuracy_and_confusion_matrix(test_path, pre_trained_weights, feature2id):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)
    correct = 0
    total = 0
    all_tags, all_preds = [],[]
    tags = list(feature2id.feature_statistics.tags)
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        total += len(sentence[2:])
        original_tags = sen[1][2:]
        pred = memm_viterbi_with_beam_search(sentence, pre_trained_weights, feature2id, beam_size=2)
        all_tags = all_tags + original_tags
        all_preds = all_preds + pred
        for i in range(len(pred)):
            if pred[i] == original_tags[i]:
                correct += 1
    print(f"Accuracy (word level): {(correct/total)*100}%")
    cm = metrics.confusion_matrix(all_tags, all_preds)
    cm_df = pd.DataFrame(cm, index = list(set(all_tags)), columns = list(set(all_tags)))
    print("confusion matrix:")
    print(cm_df.to_string())



def main():
    train_path = "train1.wtag"
    test_path = "comp1.words"

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    # print(pre_trained_weights)
    tag_all_test_with_accuracy_and_confusion_matrix('test1.wtag', pre_trained_weights, feature2id)
    # tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    # learn()
    main()
