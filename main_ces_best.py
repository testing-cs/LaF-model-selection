import argparse
import numpy as np
from config import hyperparameters
import scipy.stats as ss
from scipy.stats import spearmanr, kendalltau
import os


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
        dataTypes = ['Fog', 'original', 'Gaussian_Noise', 'Shot_Noise', 'Impulse_Noise', 'Defocus_Blur', 'Glass_Blur', 'Zoom_Blur', 'Snow', 'Brightness', 'Contrast', 'Elastic', 'JPEG', 'Pixelate', 'Frost', 'Motion_Blur']
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
            severities = range(1, 2)
        for metric in ["ces"]:
            for severity in severities:
                parameters = hyperparameters(dataName, dataType, severity)
                ground_path = parameters.save_ground_root + "ground-{0}-{1}.npy".format(dataType, severity)
                ground_accuracy = np.load(ground_path)
                ground_rank = ss.rankdata(ground_accuracy, method="min")
                iteNum = 50
                selected_modelID = np.zeros(iteNum, dtype=int)
                for ite in range(iteNum):
                    r = np.zeros((parameters.model_num, len(range(parameters.model_num, 185, 5))))
                    for numID, num in enumerate(range(parameters.model_num, 185, 5)):
                        if os.path.isfile(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite)):
                            metric_accuracy_all = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite))
                            for modelID in range(parameters.model_num):
                                metric_accuracy = metric_accuracy_all[modelID]
                                metric_rank = ss.rankdata(metric_accuracy, method="min")
                                r[modelID, numID], _ = spearmanr(ground_rank, metric_rank)
                        else:
                            # for modelID in range(parameters.model_num):
                            #     filePath = parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}-{5}.npz".format(metric, num, parameters.dataType, parameters.severity, ite, modelID)
                            #     metric_accuracy = np.load(filePath)['x']
                            #     metric_rank = ss.rankdata(metric_accuracy, method="min")
                            #     r[modelID, numID], _ = spearmanr(ground_rank, metric_rank)
                            metric_accuracy_all = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite-1))
                            for modelID in range(parameters.model_num):
                                metric_accuracy = metric_accuracy_all[modelID]
                                metric_rank = ss.rankdata(metric_accuracy, method="min")
                                r[modelID, numID], _ = spearmanr(ground_rank, metric_rank)
                    mean_r = np.mean(r, axis=1)
                    selected_modelID[ite] = np.argmax(mean_r)
                for num in range(parameters.model_num, 185, 5):
                    r = np.zeros(iteNum)
                    p = np.zeros(iteNum)
                    jac = np.zeros((iteNum, len([1, 3, 5, 10])))
                    ken = np.zeros(iteNum)
                    pken = np.zeros(iteNum)
                    for ite in range(iteNum):
                        modelID = selected_modelID[ite]
                        if os.path.isfile(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite)):
                            metric_accuracy_all = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite))
                            metric_accuracy = metric_accuracy_all[modelID]
                        else:
                            # metric_accuracy = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}-{5}.npz".format(metric, num, parameters.dataType, parameters.severity, ite, modelID))['x']
                            metric_accuracy_all = np.load(parameters.save_log_root_test + "{0}-{1}-{2}-{3}-{4}.npy".format(metric, num, parameters.dataType, parameters.severity, ite-1))
                            metric_accuracy = metric_accuracy_all[modelID]
                        ground_rank = ss.rankdata(ground_accuracy, method="min")
                        metric_rank = ss.rankdata(metric_accuracy, method="min")
                        r[ite], p[ite] = spearmanr(ground_rank, metric_rank)
                        ken[ite], pken[ite] = kendalltau(ground_rank, metric_rank)
                        ks = [1, 3, 5, 10]
                        for k_no, k in enumerate(ks):
                            ground_M = np.where(ground_rank <= k)[0]
                            metric_M = np.where(metric_rank <= k)[0]
                            jac[ite, k_no] = len(np.intersect1d(ground_M, metric_M)) / len(np.union1d(ground_M, metric_M))
                    np.save(parameters.save_result_root + "r-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), r)
                    np.save(parameters.save_result_root + "p-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), p)
                    np.save(parameters.save_result_root + "jac-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), jac)
                    np.save(parameters.save_result_root + "ken-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), ken)
                    np.save(parameters.save_result_root + "pken-{0}-{1}-{2}-{3}.npy".format(metric, num, parameters.dataType, parameters.severity), pken)


if __name__ == "__main__":
    main()
