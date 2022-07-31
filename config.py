import os


class hyperparameters():
    def __init__(self, dataName, dataType="original", severity=0):
        self.dataName = dataName
        self.dataType = dataType
        self.severity = severity
        if dataName == "fashion":
            self.class_num = 10
            self.batch_size = 100
            self.model_num = 25
        elif dataName == "mnist":
            self.class_num = 10
            self.batch_size = 100
            self.model_num = 30
        elif "mnist-" in dataName:
            self.class_num = 10
            self.batch_size = 100
            self.model_num = 10
        elif dataName == "cifar10":
            self.class_num = 10
            self.batch_size = 100
            self.model_num = 30
        elif dataName == "java250":
            self.class_num = 250
            self.batch_size = 200
            self.model_num = 20
        elif dataName == "cpp1000":
            self.class_num = 1000
            self.batch_size = 512
            self.model_num = 20
        elif dataName == "iwildcamO":
            self.class_num = 182
            self.batch_size = 100
            self.model_num = 20
        elif dataName == "amazon":
            self.class_num = 5
            self.batch_size = 100
            self.model_num = 20

        root_path = "/Volumes/1T/dnnCompare/"
        self.save_model_root = root_path + "{0}/savedM/".format(dataName)
        self.save_model_pre_root = root_path + "{0}/savedP/{1}/".format(dataName, dataType)
        self.save_log_root_test = root_path + "{0}/savedL/{1}/".format(dataName, dataType)
        self.save_result_root = root_path + "{0}/savedR/{1}/".format(dataName, dataType)
        self.save_ground_root = root_path + "{0}/savedG/{1}/".format(dataName, dataType)
        self.save_data_root_adv = root_path + "{0}/savedD/".format(dataName)
        if not os.path.isdir(self.save_model_root):
            os.makedirs(self.save_model_root)
        if not os.path.isdir(self.save_log_root_test):
            os.makedirs(self.save_log_root_test)
        if not os.path.isdir(self.save_data_root_adv):
            os.makedirs(self.save_data_root_adv)
        if not os.path.isdir(self.save_ground_root):
            os.makedirs(self.save_ground_root)
        if not os.path.isdir(self.save_result_root):
            os.makedirs(self.save_result_root)
        if not os.path.isdir(self.save_model_pre_root):
            os.makedirs(self.save_model_pre_root)
