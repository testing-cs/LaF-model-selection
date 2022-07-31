import numpy.matlib
import numpy as np
import random
import sys
from scipy import stats


def dataSel(parameters, metric, num, ite):
    global acc_list, select_index
    label_list = np.load(parameters.save_ground_root + "labels-{0}-{1}.npy".format(parameters.dataType, parameters.severity))
    if metric == "random":
        acc_list, select_index = randomSel(parameters, label_list, num)
    elif metric == "sds":
        acc_list, select_index = sdsSel(parameters, label_list, num)
    elif metric == "ces":
        cesSel(parameters, label_list, num, ite)
    else:
        sys.exit(1)

    if metric in ["random", "sds"]:
        np.savez(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npz".format(metric, num, parameters.dataType, parameters.severity, ite), x=acc_list, y=select_index)


def computeAcc(parameters, label_list, select_index, num):
    acc_list = np.zeros(parameters.model_num)
    for j in range(parameters.model_num):
        acc_list[j] = np.sum(label_list[select_index, j + 1] == label_list[select_index, 0]) / num
    return acc_list


def randomSel(parameters, label_list, num):
    select_index = random.sample(np.arange(label_list.shape[0]).tolist(), num)
    acc_list = computeAcc(parameters, label_list, select_index, num)
    return acc_list, select_index


def select_random(parameters, item_dis_rank, label_list, num):
    rank_test = item_dis_rank[:int(len(item_dis_rank) * 0.25)].tolist()
    select_index = random.sample(rank_test, num)
    acc_list = computeAcc(parameters, label_list, select_index, num)
    return acc_list, select_index


def label_compare(label_list):
    mode_list, _ = stats.mode(np.transpose(label_list), axis=0)
    return np.transpose(mode_list).astype(int)


def r_rate(label_list, mode_list):
    model_rate = np.sum(label_list == numpy.matlib.repmat(mode_list, 1, label_list.shape[1]), axis=0)
    return np.argsort(model_rate)[::-1], model_rate


def item_discrimination(r_rank, label_list, mode_list):
    top_rank = r_rank[:int(len(r_rank) * 0.27)]
    last_rank = r_rank[int(len(r_rank) * 0.73):]
    score1 = np.sum(label_list[:, top_rank] == numpy.matlib.repmat(mode_list, 1, len(top_rank)), axis=1) / len(top_rank)
    score2 = np.sum(label_list[:, last_rank] == numpy.matlib.repmat(mode_list, 1, len(last_rank)), axis=1) / len(last_rank)
    item_dis = score1 - score2
    return np.argsort(item_dis)[::-1], item_dis


def item_difficulty(label_list, mode_list):
    item_dif = np.sum(label_list == numpy.matlib.repmat(mode_list, 1, label_list.shape[1]), axis=1) / label_list.shape[1]
    return item_dif


def sdsSel(parameters, label_list, num):
    # step 2: vote for estimated labels
    mode_list = label_compare(label_list[:, 1:])
    # step 3: classify top/bottom models
    r_rank, _ = r_rate(label_list[:, 1:], mode_list)
    # step 4: compute sample discrimination
    item_dis_rank, _ = item_discrimination(r_rank, label_list[:, 1:], mode_list)
    # step 5: randomly select sample from top 25%
    acc_list, select_index = select_random(parameters, item_dis_rank, label_list, num)

    return acc_list, select_index


def conditional_sample(model_pre, num, test_size):
    batch_size = 5
    iterate = int((num - 10)/batch_size)
    select_index = np.random.choice(range(test_size), replace=False, size=10)
    candidate_index = list(set(range(test_size)) - set(select_index))
    for i in range(iterate):
        min_ce = np.inf
        for j in range(30):
            select_group_index = np.random.choice(candidate_index, replace=False, size=batch_size)
            group_p = build_neuron_tables(model_pre, select_group_index)
            pool_p = build_neuron_tables(model_pre, select_index)
            pool_p_log = np.log2(pool_p)
            pool_p_log[pool_p_log == -np.inf] = 0
            ce_current = -np.sum(np.multiply(group_p, pool_p_log))
            if ce_current < min_ce:
                min_ce = ce_current
                group_select = select_group_index
        select_index = np.append(select_index, group_select)
        candidate_index = list(set(candidate_index) - set(group_select))

    return select_index


def build_neuron_tables(model_pre, set_index):
    model_neuron = np.zeros((model_pre.shape[1], 5))
    for class_no in range(model_pre.shape[1]):
        max_val = np.max(model_pre[set_index, class_no], axis=0)
        min_val = np.min(model_pre[set_index, class_no], axis=0)
        interval = np.linspace(min_val, max_val, 5)
        for interval_id in range(4):
            model_neuron[class_no, interval_id] = np.sum(np.logical_and(model_pre[set_index, class_no] >= interval[interval_id], model_pre[set_index, class_no] < interval[interval_id + 1]))
        model_neuron[class_no, 4] = len(set_index) - np.sum(model_neuron[class_no, :4])

    return (model_neuron/ len(set_index)).flatten()


def cesSel(parameters, label_list, num, ite):
    acc_all = []
    for modelID in range(parameters.model_num):
        model_pre = np.load(parameters.save_model_pre_root + "pre-{0}-{1}-{2}.npy".format(modelID, parameters.dataType, parameters.severity))
        select_index = conditional_sample(model_pre, num, label_list.shape[0])
        acc_list = computeAcc(parameters, label_list, select_index, num)
        acc_all.append(acc_list)
    np.save(parameters.save_log_root_test + "ces-{0}-{1}-{2}-{3}.npy".format(num, parameters.dataType, parameters.severity, ite, ), acc_all)
