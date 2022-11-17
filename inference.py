from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy as np


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    MEMM Viterbi algorithm implementation without beam search
    :param sentence: sentence to predict
    :param pre_trained_weights: weights of the model
    :param feature2id: class that contains the features and the tags
    :return: predicted tags
    """
    n = len(sentence) - 1
    sentence = sentence[:n]
    tags = feature2id.feature_statistics.tags
    tags_pred = [""] * n
    pi, bp, _ = init_pi_bp_beam_dict(n, tags)
    for k in range(2, n):
        if k == 2:
            u = "*"  # the first tag
            t = "*"  # tag of the word before the first word
            for v in tags:  # iterating over the tags of the current word
                q_value = q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags)  # q(v|t,u)
                current = pi[(k - 1, t, u)] * q_value  # calculation of current probability
                pi[(k, u, v)] = current
                bp[(k, u, v)] = t
        else:
            for u in tags:  # iterating over the tags of the previous word
                for v in tags:  # iterating over the tags of the current word
                    max_prob = 0  # initialization of the max probability
                    arg_max_prob = ""  # initialization of the label of the max probability
                    for t in tags:  # iterating over the tags of the word before the previous word
                        q_value = q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags)  # q(v|t,u)
                        current = pi[(k - 1, t, u)] * q_value  # calculation of current probability

                        if current > max_prob:  # updating the max probability and the label of the max probability
                            max_prob = current
                            arg_max_prob = t

                    pi[(k, u, v)] = max_prob  # updating pi
                    bp[(k, u, v)] = arg_max_prob  # updating bp

    max_u_v = find_max_u_v(pi, n)
    (tags_pred[n - 2], tags_pred[n - 1]) = (max_u_v[0], max_u_v[1])
    for k in range(n - 3, -1, -1):
        tags_pred[k] = bp[(k + 2, tags_pred[k + 1], tags_pred[k + 2])]

    final_pred = tags_pred[2:] + ["~"]
    return final_pred


def init_pi_bp_beam_dict(n, tags):
    """
    initialization of pi, bp and beam search dictionaries
    :param n: length of the sentence
    :param tags: known tags
    :return: pi, bp and beam search dictionaries
    """
    pi = {}
    bp = {}
    beam_dict = {}
    for i in range(n):
        for u in tags:
            for v in tags:
                pi[(i, u, v)] = 0
                bp[(i, u, v)] = ""

    pi[(1, "*", "*")] = 1
    beam_dict[1] = ["*"]
    return pi, bp, beam_dict


def find_max_u_v(pi, n):
    """
    finding the max u and v
    :param pi: pi dictionary
    :param n: length of the sentence
    :return: u and v such that pi[(n-1, u, v)] is the max value
    """
    max_u_v = ("", "")
    max_num = 0
    for key in pi.keys():
        if key[0] == n - 1:
            if pi[key] > max_num:
                max_num = pi[key]
                max_u_v = (key[1], key[2])
    return max_u_v


def memm_viterbi_with_beam_search(sentence, pre_trained_weights, feature2id, beam_size):
    """
    MEMM Viterbi algorithm implementation with beam search
    :param sentence: sentence to predict
    :param pre_trained_weights: weights of the model
    :param feature2id: class that contains the features and the tags
    :return: predicted tags
    """
    n = len(sentence)-1  # length of the sentence without the last word ("~")
    sentence = sentence[:n]  # sentence without the last word ("~")
    tags = feature2id.feature_statistics.tags  # known tags
    tags_pred = [""] * n  # initialization of the predicted tags list
    pi, bp, beam_dict = init_pi_bp_beam_dict(n, tags)  # initialization of pi, bp and beam search dictionaries
    for k in range(2, n):
        list_of_probs = []
        if k == 2:
            u = "*"  # the first tag
            t = "*"  # tag of the word before the first word
            for v in tags: # iterating over the tags of the current word
                q_value = q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags)  # q(v|t,u)
                current = pi[(k - 1, t, u)] * q_value  # calculation of current probability
                pi[(k, u, v)] = current
                bp[(k, u, v)] = t
                list_of_probs.append((current, v))
            max_probs_and_labels = sorted(list_of_probs)[-beam_size:]  # pruning with beam search
            beam_dict[k] = [x[1] for x in max_probs_and_labels]  # saving the labels of the beam
        else:
            for u in beam_dict[k-1]:  # iterating over the tags of the previous word according to the beam search
                for v in tags:  # iterating over the tags of the current word
                    max_prob = 0  # initialization of the max probability
                    arg_max_prob = ""  # initialization of the label of the max probability
                    for t in beam_dict[k-2]:  # iterating over the tags of the word before the previous word

                        q_value = q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags)  # q(v|t,u)
                        current = pi[(k - 1, t, u)] * q_value  # calculation of current probability

                        if current > max_prob:  # updating the max probability and the label of the max probability
                            max_prob = current
                            arg_max_prob = t

                    pi[(k, u, v)] = max_prob  # updating pi
                    bp[(k, u, v)] = arg_max_prob  # updating bp
                    list_of_probs.append((max_prob, v))  # saving the probabilities and the labels

            beam_dict[k] = [x[1] for x in sorted(list_of_probs)[-beam_size:]]  # pruning with beam search

    max_u_v = find_max_u_v(pi, n)
    tags_pred[n - 2] = max_u_v[0]
    tags_pred[n - 1] = max_u_v[1]
    for k in range(n - 3, -1, -1):
        tags_pred[k] = bp[(k + 2, tags_pred[k + 1], tags_pred[k + 2])]

    final_pred = tags_pred[2:] + ["~"]
    return final_pred

def q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags):
    """
    q function implementation
    """
    # building history of :"(c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)"
    nominator = 0
    denominator = 0
    # calculating the nominator and the denominator
    for tag in tags:
        if k == 1:
            history = (sentence[k], tag, sentence[k - 1], u, "*", t, sentence[k + 1])
        elif k == len(sentence) - 1:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, "~")
        else:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])

        if history not in feature2id.histories_features.keys():
            features = represent_input_with_features(history, feature2id.feature_to_idx)
        else:
            features = feature2id.histories_features[history]
        current_calc = np.exp(sum([pre_trained_weights[feature] for feature in features]))
        if tag == v:
            nominator = current_calc
        denominator += current_calc

    return nominator / denominator


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)
    output_file = open(predictions_path, "a+")
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi_with_beam_search(sentence, pre_trained_weights, feature2id, beam_size=2)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


def create_indices_of_tags(tags):
    tag_to_index = {}
    for index, tag in enumerate(tags):
        tag_to_index[tag] = index
    return tag_to_index

#not in use
def features_list_to_binary_vector(features_indices, total_features):
    binary_vector = np.zeros(total_features)
    for feature_idx in features_indices:
        binary_vector[feature_idx] = 1
    return binary_vector
