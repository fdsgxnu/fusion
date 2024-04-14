import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from ifm_res_fca_train import ECG
from ifm_res_fca_model import res_ifm


# 混淆矩阵横轴的和为实际数量，对应的竖轴为预测的类别，斜线为正确的预测结果
class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()  # print直接输出表格
        table.field_names = ["", "ACC", "PPV", "SE", "SP"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            # acc正确率
            ACC = round((TN + TP) / (TP + FN + FP + TN), 3) if TP + FN + FP + TN != 0 else 0.
            # ppv阳性预测值
            PPV = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            # se灵敏度
            SE = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            # sp特异性
            SP = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], ACC, PPV, SE, SP])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据读取
    test_data = np.load('D:/pythonProject/data_mit/new_classification/test_data_224_5.npy')  # caution,not new_classification1
    test_label = np.load('D:/pythonProject/data_mit/new_classification/test_label_224_5.npy')

    # 数据升维，标签转换独热码
    test_data = np.expand_dims(test_data, axis=2)
    test_label = torch.eye(5)[test_label, :]

    a = ECG(test_data, test_label)  # train中的ECG转换类型函数
    test_num = len(a)
    print('using {} data for testing'.format(test_num))

    test_Loader = DataLoader(a, batch_size=64, num_workers=0)
    net = res_ifm()
    net.to(device)

    net.load_state_dict(torch.load('calculate_para.pth',
                                   map_location=torch.device("cuda:0")), strict=False)  # 训练保存的模型
    acc = 0.0

    # read class_indict
    json_label_path = './class_indices.json'  # 读取分类类别文件
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    i = 0
    with torch.no_grad():
        for data_test in test_Loader:
            v1, v_table = data_test
            v1 = v1.type(torch.FloatTensor)
            v1 = torch.squeeze(v1, -1)

            output = net(v1.to(device))
            val_label = torch.topk(v_table, 1)[1].squeeze(1)
            output = torch.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            confusion.update(output.to("cpu").numpy(), val_label.to("cpu").numpy())
            # i += 1
            # print(i)
    confusion.plot()
    confusion.summary()
