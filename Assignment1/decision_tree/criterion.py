"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    # =============== TODO (students) ===============
    num_parent = len(y)
    num_left = len(l_y)
    num_right = len(r_y)

    # Entropy H = -sum(p * log2(p)).
    # An empty child has no labels, so its sum is simply 0.
    parent_entropy = -sum(count / num_parent * math.log2(count / num_parent)
                          for count in all_labels.values())
    left_entropy = -sum(count / num_left * math.log2(count / num_left)
                        for count in left_labels.values())
    right_entropy = -sum(count / num_right * math.log2(count / num_right)
                         for count in right_labels.values())

    # Children entropy is weighted by |child| / |parent|.
    children_entropy = (num_left * left_entropy
                        + num_right * right_entropy) / num_parent
    info_gain = parent_entropy - children_entropy
    # ===============================================
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    # =============== TODO (students) ===============
    num_parent = len(y)

    # Split info is the entropy of the left/right sample proportions.
    # An empty side is skipped, since 0 * log2(0) is taken as 0.
    split_info = -sum(num / num_parent * math.log2(num / num_parent)
                      for num in (len(l_y), len(r_y)) if num > 0)

    # split_info == 0 means one side is empty, i.e. a useless split.
    info_gain = info_gain / split_info if split_info > 0 else 0.0
    # ===============================================
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    num_parent = len(y)
    num_left = len(l_y)
    num_right = len(r_y)

    # Gini = 1 - sum(p^2).
    # An empty child gets Gini 1 here, but its weight below is 0.
    parent_gini = 1 - sum((count / num_parent) ** 2 for count in all_labels.values())
    left_gini = 1 - sum((count / num_left) ** 2 for count in left_labels.values())
    right_gini = 1 - sum((count / num_right) ** 2 for count in right_labels.values())

    before = parent_gini
    # Children Gini is weighted by |child| / |parent|.
    after = (num_left * left_gini + num_right * right_gini) / num_parent
    # ===============================================
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    # =============== TODO (students) ===============
    num_parent = len(y)
    num_left = len(l_y)
    num_right = len(r_y)

    # Error E = 1 - max(p), i.e. the share of samples outside the majority class.
    before = (num_parent - max(all_labels.values())) / num_parent

    # Weighted children error = misclassified samples in both children / |parent|.
    # default=0 handles an empty child (no samples, no errors).
    left_errors = num_left - max(left_labels.values(), default=0)
    right_errors = num_right - max(right_labels.values(), default=0)
    after = (left_errors + right_errors) / num_parent
    # ===============================================
    return before - after
