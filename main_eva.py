import argparse
import numpy as np
from config import hyperparameters
import scipy.stats as ss
from scipy.stats import spearmanr, kendalltau


def Jaccard_similarity(u, v):
    up = np.intersect1d(u, v)
    down = np.union1d(u, v)
    jac = len(up) / len(down)
    return jac


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='java250', type=str, choices=['mnist', 'cifar10', 'fashion', 'amazon', 'iwildcamO', 'java250', 'C++1000'])
    return parser.parse_args()


def main():
    global _, ground_accuracy, metric_accuracy
    args = get_args()
    dataName = args.dataName
    if dataName in ['mnist', 'cifar10']:
        dataTypes = ['original', 'Gaussian_Noise', 'Shot_Noise', 'Impulse_Noise', 'Defocus_Blur', 'Glass_Blur',
                     'Zoom_Blur', 'Snow', 'Fog', 'Brightness', 'Contrast', 'Elastic', 'JPEG', 'Pixelate', 'Frost', 'Motion_Blur']
    elif dataName in ["fashion", "cpp1000"]:
        dataTypes = ['original']
    elif dataName == "java250":
        dataTypes = ['original', 'ood']
    else:
        dataTypes = ["id", "ood"]
    for dataType in dataTypes:
        if dataType in ["original", "id", "ood"]:
            severities = [0]
        else:
            severities = range(1, 6)
        for metric in ["sds", "random", "new"]:  # "new" is laf
            for severity in severities:
                parameters = hyperparameters(dataName, dataType, severity)
                ground_path = parameters.save_ground_root + "ground-{0}-{1}.npy".format(dataType, severity)
                ground_accuracy = np.load(ground_path)
                if metric in ["random", "sds"]:
                    iteNum = 50
                else:
                    iteNum = 1
                if metric in ['new']:
                    metric_accuracy = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npz".format(metric, 0, parameters.dataType, parameters.severity, 0))['x']
                    ground_rank = ss.rankdata(ground_accuracy, method="min")
                    metric_rank = ss.rankdata(metric_accuracy, method="min")
                    print(f"{dataType}-{31 - metric_rank}")
                    r, p = spearmanr(ground_rank, metric_rank)
                    jac = np.zeros((1, len([1, 3, 5, 10])))
                    ken, pken = kendalltau(ground_rank, metric_rank)
                    ks = [1, 3, 5, 10]
                    for k_no, k in enumerate(ks):
                        ground_M = np.where(len(ground_rank) + 1 - ground_rank <= k)[0]
                        metric_M = np.where(len(ground_rank) + 1 - metric_rank <= k)[0]
                        jac[0, k_no] = len(np.intersect1d(ground_M, metric_M)) / len(np.union1d(ground_M, metric_M))
                    np.save(parameters.save_result_root + "r-{0}-{1}-{2}-{3}.npy".format(metric, 0, parameters.dataType, parameters.severity), r)
                    np.save(parameters.save_result_root + "p-{0}-{1}-{2}-{3}.npy".format(metric, 0, parameters.dataType, parameters.severity), p)
                    np.save(parameters.save_result_root + "jac-{0}-{1}-{2}-{3}.npy".format(metric, 0, parameters.dataType, parameters.severity), jac)
                    np.save(parameters.save_result_root + "ken-{0}-{1}-{2}-{3}.npy".format(metric, 0, parameters.dataType, parameters.severity), ken)
                    np.save(parameters.save_result_root + "pken-{0}-{1}-{2}-{3}.npy".format(metric, 0, parameters.dataType, parameters.severity), pken)
                else:
                    for num in range(parameters.model_num, 185, 5):
                        r = np.zeros(iteNum)
                        p = np.zeros(iteNum)
                        jac = np.zeros((iteNum, len([1, 3, 5, 10])))
                        ken = np.zeros(iteNum)
                        pken = np.zeros(iteNum)
                        for ite in range(iteNum):
                            metric_accuracy = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npz".format(metric, num, parameters.dataType, parameters.severity, ite))['x']
                            ground_rank = ss.rankdata(ground_accuracy, method="min")
                            metric_rank = ss.rankdata(metric_accuracy, method="min")
                            r[ite], p[ite] = spearmanr(ground_rank, metric_rank)
                            ken[ite], pken[ite] = kendalltau(ground_rank, metric_rank)
                            ks = [1, 3, 5, 10]
                            for k_no, k in enumerate(ks):
                                ground_M = np.where(len(ground_rank) + 1 - ground_rank <= k)[0]
                                metric_M = np.where(len(ground_rank) + 1 - metric_rank <= k)[0]
                                jac[ite, k_no] = len(np.intersect1d(ground_M, metric_M)) / len(np.union1d(ground_M, metric_M))
                        np.save(parameters.save_result_root + "r-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), r)
                        np.save(parameters.save_result_root + "p-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), p)
                        np.save(parameters.save_result_root + "jac-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), jac)
                        np.save(parameters.save_result_root + "ken-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), ken)
                        np.save(parameters.save_result_root + "pken-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), pken)


if __name__ == "__main__":
    main()
