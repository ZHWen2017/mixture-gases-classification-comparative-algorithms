from sklearn.metrics import classification_report,confusion_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

result_path = 'test_result_3method.txt'
result_path1='test_result_ANN.txt'
result_path2='test_result_1D-DCNN.txt'
f1 = open(result_path, 'rb')
result_3m = pickle.load(f1)
f1.close()
f2 = open(result_path1, 'rb')
result_ANN = pickle.load(f2)
f2.close()
f3 = open(result_path2, 'rb')
result_CNN = pickle.load(f3)
f3.close()

####绘制混淆矩阵图
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    # fig = plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize='small')
    plt.yticks(tick_marks, classes, fontsize='small')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    override = {
        'fontsize': 'large',
        'verticalalignment': 'top',
        'horizontalalignment': 'center',
    }
    plt.ylabel('True Label', override,labelpad=0.5)
    plt.xlabel('Predicted Label',override,labelpad=0.5)
    save_path = 'C:/Users/Administrator/Desktop/对比算法/result_fig/'
    plt.savefig(title+'.eps')
    plt.show()

result_3m['ANN'] = result_ANN['ANN']; result_3m['1D-DCNN'] = result_CNN['1D-DCNN']
result_all = result_3m
con_mats = {};   labels = list(set(result_all['ANN'][0]))
for method, ys in result_all.items():
    y_true = ys[0]; y_predict = ys[1]

    targetNames = ['Air', 'Eth', 'CO', 'Met', 'Eth-CO','Eth-Met']
    print('%s calculating result\n'% method)
    class_report = classification_report(y_true,y_predict, labels=labels, target_names=targetNames)
    print(class_report)
    con_mat = confusion_matrix(y_true,y_predict,labels=labels)


    con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    con_mats[method] = con_mat
    plot_confusion_matrix(con_mat,classes=targetNames,normalize=True,title='%s Confusion Matrix'%method)
# print(con_mats)
###绘制混淆矩阵图

