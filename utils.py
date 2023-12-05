import collections
import matplotlib
import matplotlib.pyplot as plt

def get_data(file_name):
    a = open(file_name, 'r')
    train_loss = collections.defaultdict(float)
    train_acc = collections.defaultdict(float)
    val_loss = collections.defaultdict(float)
    val_acc = collections.defaultdict(float)
    test_loss = 0
    test_acc = 0
    count = 0
    for line in a:
        line = line.split()
        if line[1] == 'train':
            train_loss[int(line[0])] += float(line[2])
            train_acc[int(line[0])] += float(line[3])
        elif line[1] == 'val':
            val_loss[int(line[0])] += float(line[2])
            val_acc[int(line[0])] += float(line[3])
        elif line[1] == 'test':
            test_loss += float(line[2])
            test_acc += float(line[3])
            count += 1
        else:
            print("error")
    a.close()
    for key in train_loss.keys():
        train_loss[key] /= count
        train_acc[key] /= count
        val_loss[key] /= count
        val_acc[key]  /= count
    test_loss /= count
    test_acc /= count

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


# test_loss和test_acc也跟train以及validation一样的格式
def get_data_new(file_name):
    a = open(file_name, 'r')
    train_loss = collections.defaultdict(float)
    train_acc = collections.defaultdict(float)
    val_loss = collections.defaultdict(float)
    val_acc = collections.defaultdict(float)
    test_loss = collections.defaultdict(float)
    test_acc = collections.defaultdict(float)
    count = 0
    for line in a:
        line = line.split()
        if line[1] == 'train':
            train_loss[int(line[0])] += float(line[2])
            train_acc[int(line[0])] += float(line[3])
        elif line[1] == 'val':
            val_loss[int(line[0])] += float(line[2])
            val_acc[int(line[0])] += float(line[3])
        elif line[1] == 'test':
            test_loss[int(line[0])] += float(line[2])
            test_acc[int(line[0])] += float(line[3])
        else:
            print("error")
    a.close()
    # for key in train_loss.keys():
    #     train_loss[key] /= count
    #     train_acc[key] /= count
    #     val_loss[key] /= count
    #     val_acc[key] /= count
    # test_loss /= count
    # test_acc /= count

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


def get_plot(file_name):
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = get_data(file_name)
    
    #plt.plot(train_loss.keys(), train_loss.values(), label = 'train_loss')
    #print(train_acc.values())
    plt.plot(train_acc.keys(), train_acc.values(), label = 'train_acc')
    #plt.plot(val_loss.keys(), val_loss.values(), label = 'val_loss')
    plt.plot(val_acc.keys(), val_acc.values(), label = 'val_acc')
    plt.ylim(0, 10)
    plt.xlim(0, 500)
    plt.legend()
    plt.show()

def plot_training_accuracy(file_name):
    # naming the x axis
    plt.xlabel('Number of Layers', fontsize=25)
    # naming the y axis
    plt.ylabel('Accuracy', fontsize=25)

    plt.legend(fontsize=12, frameon=True)  # , loc='lower right')
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.savefig("train_accuracy.eps", format='eps')
    # %%
    plt.style.use('seaborn-whitegrid')
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = get_data(file_name)
    plt.plot(val_acc.keys(), val_acc.values(), 'o-', label='-layer', markevery=10, markersize=8)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# fig_para: 'train', 'val', 'test'
def plot_training_accuracy(file_name_list, lengend_list, fig_para):
    result_list = []
    for file_name in file_name_list:
        # result的结构train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
        result = get_data_new(file_name)
        result_list.append(result)

    fig_para_list = ['train', 'val', 'test']
    fig_para_index = fig_para_list.index(fig_para)
    y_label_list = ['train_acc', 'val_acc', 'test_acc']

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    matplotlib.rc('font', size=24)

    for i in range(len(file_name_list)):
        acc = result_list[i][fig_para_index*2 + 1]  # fig_para_index:loss, fig_para_index+1:acc
        plt.plot(list(acc.keys())[:100], list(acc.values())[:100], label=lengend_list[i], linewidth=2)
        print(file_name_list[i], max(list(acc.values())[100:200]))

    plt.xlabel('Global Epoch ($\#$)', fontdict={'weight': 'normal'})  # X轴标签
    plt.ylabel(y_label_list[fig_para_index], fontdict={'weight': 'normal'})  # Y轴标签

    plt.legend(loc='best')
    plt.grid()
    # plt.savefig('../results/fig_gpu_utilization.pdf', format='pdf', bbox_inches='tight')
    plt.show()


filename = '1cora_IID_centralized_2layer_GCN_iter_200'
filename1 = 'cora_IID_centralized_3layer_GCN_iter_200'
# filename2 = 'cora_IID_centralized_mini_2layer_GCN_iter_200'
filename2 = '1cora_IID_0.0_0hop_Block_federated_3layer_GCN_iter_200_epoch_3_device_num_7'
filename3 = 'cora_IID_0.0_2hop_Block_federated_3layer_GCN_iter_200_epoch_3_device_num_7'
filename4 = 'cora_IID_centralized_cluster_mini_3layer_GCN_iter_200'
# get_plot(filename1)
# get_plot(filename2)

# plot_training_accuracy(filename2)

file_name_list = [filename, filename1, filename2, filename3, filename4]
# 默认是三层layers
lengend_list = ['GCN-2layer','GCN','FedGCN-noC','FedGCN-fullC', 'GCN-cluster-noC', 'FGL-2hop']
fig_para = 'val'  # 取值'train','val','test'
# plot_training_accuracy(file_name_list[:5], lengend_list[:5], fig_para)


# ###部分clients参与training

fn1 = 'cora_IID_1_0hop_federated_cluster_partition_partial_7_2layer_GCN_iter_200_epoch_3_device_num_7'
fn2 = 'cora_IID_1_0hop_federated_cluster_partition_partial_5_2layer_GCN_iter_200_epoch_3_device_num_7'
fn3 = 'cora_IID_1_0hop_federated_cluster_partition_partial_3_2layer_GCN_iter_200_epoch_3_device_num_7'
fn_list = [fn1, fn2, fn3]

lengend_list1 = ['K=7(Full)','K=5','K=3']

# plot_training_accuracy(fn_list, lengend_list1, fig_para)

# 测试history emb.对training的影响

fn1 = 'ora_IID_1_2hop_Block_federated_2layer_GCN_iter_200_epoch_3_device_num_7'
fn2 = 'ora_IID_1_2hop_federated_embedding_realtime_2layer_GCN_iter_200_epoch_3_device_num_7'
fn3 = 'ora_IID_1_2hop_federated_embedding_periodic_1_2layer_GCN_iter_200_epoch_3_device_num_7'
fn4 = 'ora_IID_1_2hop_federated_embedding_periodic_10_2layer_GCN_iter_200_epoch_3_device_num_7'
fn5 = 'ora_IID_1_2hop_federated_embedding_periodic_50_2layer_GCN_iter_200_epoch_3_device_num_7'

fn_list = [fn1, fn3, fn4, fn5]

lengend_list1 = ['features','frequency = 1','frequency = 10','frequency = 50']

# plot_training_accuracy(fn_list, lengend_list1, fig_para)

f1 = 'ogbn-arxiv_IID_centralized_2layer_GCN_iter_200'
f2 = 'gbn-arxiv_IID_1_0hop_Block_federated_2layer_GCN_iter_200_epoch_3_device_num_40'

fn_list = [f1, f2]

lengend_list = ['centralizedGCN', 'FedGCN-noC']

# plot_training_accuracy(fn_list, lengend_list, fig_para)

# f1 = 'cora_IID_centralized_2layer_GCN_iter_200'
# f2 = 'cora_IID_1_0hop_Block_federated_2layer_GCN_iter_200_epoch_3_device_num_7'
f3 = 'cora_IID_1_0hop_federated_cluster_2layer_GCN_iter_200_epoch_3_device_num_7'
f5 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_1_2layer_GCN_iter_200_epoch_3_device_num_7'
f6 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_40_2layer_GCN_iter_200_epoch_3_device_num_7'
f7 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_300_2layer_GCN_iter_200_epoch_3_device_num_7'
fn_list = [f1, f2, f3,  f5, f6, f7]

lengend_list = ['centralizedGCN', 'FedGCN-noC', 'FedGCN-clusterpartition',
                 'new-1', 'new-40', 'new-300']

# plot_training_accuracy(fn_list[2:], lengend_list[2:], fig_para)

f1 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_update_1_2layer_GCN_iter_200_epoch_3_device_num_7'
f2 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_1_2layer_GCN_iter_200_epoch_3_device_num_7'
f3 = 'cora_IID_1_2hop_federated_embedding_periodic_cluster_25_2layer_GCN_iter_200_epoch_3_device_num_7'
f4 = 'citeseer_IID_1_2hop_federated_cluster_2layer_GCN_iter_200_epoch_3_device_num_6'
fn_list = [f1, f2, f3]

lengend_list = ['mixed-frequency', 'frequency=1', 'frequency=25',
                'FedGCN-cluster-2hop', 'new-40', 'new-300']

# plot_training_accuracy(fn_list, lengend_list, fig_para)

f1 = 'cora_IID_centralized_2layer_GCN_iter_200'
f2 = 'cora_IID_centralized_cluster_2layer_GCN_iter_200'
f3 = 'ora_IID_1_0hop_3_participants_federated_cluster_partial_2layer_GCN_iter_200_epoch_3_device_num_7'

f1 = 'pubmed_IID_centralized_2layer_GCN_iter_200'
f2 = 'pubmed_IID_centralized_cluster_2layer_GCN_iter_200'
fn_list = [f1, f2]

lengend_list = ['GCN', 'GCN-cluster']

plot_training_accuracy(fn_list, lengend_list, fig_para)


