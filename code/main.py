import time
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from util.data_split import load_data, tasks_generator
from torch_geometric.utils import add_self_loops
from params import set_params
from sefsnc import SeFsnc
import warnings
from copy import deepcopy
import scipy.sparse as sp
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def get_tree_layer_weight(w, tree_height):
    if w >= 1:
        raise ValueError("w must be less than 1")
    n = torch.arange(tree_height, dtype=torch.float32)
    weights = w * (1 - w) ** n
    return weights


def get_nmi_ari(test_z, test_y, n_way, rs=0):
    pred_y = KMeans(n_clusters=n_way, random_state=rs).fit(test_z.detach().cpu().numpy()).labels_
    nmi = normalized_mutual_info_score(test_y, pred_y)
    ari = adjusted_rand_score(test_y, pred_y)
    return nmi, ari

class TrainFlow:
    def __init__(self):
        self.data = load_data(args.dataset_name, args.tree_height, args.t, args.big).to(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_weight = get_tree_layer_weight(args.init_w, args.tree_height)
        self.se_net = SeFsnc(self.data.num_feature, args.num_hidden, args.num_head, args.num_layers_gat, args.feat_drop,
                             args.attn_drop, args.negative_slope, 4, args.n_way, args.tree_height,
                             self.layer_weight, args.feat_drop, args.tau, args.big, args.batch)

        self.opti_encoder = torch.optim.Adam(self.se_net.graph_encoder.parameters(), lr=args.encoder_lr,
                                             weight_decay=args.encoder_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opti_encoder, 0.99)
        self.opti_cls = torch.optim.Adam(self.se_net.cls.parameters(), lr=args.cls_lr,
                                         weight_decay=args.cls_weight_decay)

        if torch.cuda.is_available() and args.gpu != -1:
            print('Using CUDA')
            self.se_net.cuda()
        self.best_acc_val = 0
        self.old_best_acc = 0
        self.best_test = 0
        self.best_param_c = 0
        self.best_encoder_weight = None

    def train_cls(self, id_support, id_query, class_selected):
        num_query = args.n_way * args.m_qry
        losses_q = [0 for _ in range(args.cls_update_step + 1)]
        corrects = [0 for _ in range(args.cls_update_step + 1)]

        # graph_embed = self.se_net.get_graph_embed(self.data.x, self.data.edge_index)
        graph_embed = self.se_net.get_graph_embed(self.data.graph, self.data.x)
        graph_embed = torch.cat(graph_embed, axis=1)

        # Model Agnostic Meta Learning
        for i in range(args.num_train):
            y_support = torch.tensor(
                [class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_support[i]]]).to(self.device)
            y_query = torch.tensor([class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_query[i]]]).to(
                self.device)

            logits = self.se_net.get_logits(graph_embed[id_support[i]])
            loss = F.nll_loss(logits, y_support)
            grad = torch.autograd.grad(loss, self.se_net.cls.parameters())
            fast_weights = list(
                map(lambda p: p[1] - args.cls_update_lr * p[0], zip(grad, self.se_net.cls.parameters())))

            # this is the loss and accuracy before the cls first update
            with torch.no_grad():
                logits_q = self.se_net.get_logits(graph_embed[id_query[i]])
                loss_q = F.nll_loss(logits_q, y_query)
                losses_q[0] += loss_q
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_query).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the cls first update
            with torch.no_grad():
                logits_q = self.se_net.get_logits(graph_embed[id_query[i]], fast_weights)  ##权重的问题
                loss_q = F.nll_loss(logits_q, y_query)
                losses_q[1] += loss_q
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_query).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, args.cls_update_step):
                # 1. run the i-th task and compute loss for k=1~K-1=
                logits = self.se_net.get_logits(graph_embed[id_support[i]], fast_weights)
                loss = F.nll_loss(logits, y_support)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - args.cls_update_lr * p[0], zip(grad, fast_weights)))
                # 4. compute cls loss on query set
                logits_q = self.se_net.get_logits(graph_embed[id_query[i]], fast_weights)
                loss_q = F.nll_loss(logits_q, y_query)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = logits_q.argmax(dim=1)
                    correct = torch.eq(pred_q, y_query).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        cls_loss = losses_q[-1] / args.num_train
        cls_acc = corrects[-1] / (args.num_train * num_query)
        return cls_loss, cls_acc

    def train_encoder(self, id_support, id_query, y_support, y_query):
        losses_q = [0 for _ in range(args.cls_update_step + 1)]
        graph_embed = self.se_net.get_graph_embed(self.data.graph, self.data.x)
        mi_loss = self.se_net.get_mi_loss(graph_embed, self.data.tree_partition, args.num_head, self.data.contrast_weight)
        graph_embed = torch.cat(graph_embed, axis=1)
        logits = self.se_net.get_logits(graph_embed[id_support])
        loss = F.nll_loss(logits, y_support)
        grad = torch.autograd.grad(loss, self.se_net.cls.parameters())
        fast_weights = list(
            map(lambda p: p[1] - args.cls_update_lr * p[0], zip(grad, self.se_net.cls.parameters())))

        for k in range(1, args.cls_update_step):
            # 1. run the i-th task and compute loss for k=1~K-1=
            logits = self.se_net.get_logits(graph_embed[id_support], fast_weights)
            loss = F.nll_loss(logits, y_support)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - args.cls_update_lr * p[0], zip(grad, fast_weights)))
            # 4. compute cls loss on query set
            logits_q = self.se_net.get_logits(graph_embed[id_query], fast_weights)
            loss_q = F.nll_loss(logits_q, y_query)
            losses_q[k + 1] += loss_q

        cls_loss = losses_q[-1]
        return mi_loss, cls_loss

    def fs_test(self, id_support, id_query, class_selected, param_c, solver='lbfgs', iter_clf=700, output_ari=False):
        graph_embed = self.se_net.get_graph_embed(self.data.graph, self.data.x)
        graph_embed = torch.cat(graph_embed, axis=1)
        test_acc_all = []
        nmi_all = 0
        ari_all = 0
        for i in range(args.num_test):
            train_z = graph_embed[id_support[i]]
            test_z = graph_embed[id_query[i]]
            train_y = np.array([class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_support[i]]])
            test_y = np.array([class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_query[i]]])
            clf = LogisticRegression(solver=solver, max_iter=iter_clf, multi_class='auto',
                                     C=param_c).fit(train_z.detach().cpu().numpy(), train_y)
            test_acc = clf.score(test_z.detach().cpu().numpy(), test_y)
            test_acc_all.append(test_acc)

            if output_ari:
                nmi, ari = get_nmi_ari(test_z, test_y, args.n_way)
                nmi_all += nmi / args.num_test
                ari_all += ari / args.num_test

        final_mean = np.mean(test_acc_all)
        final_std = np.std(test_acc_all)
        final_interval = 1.96 * (final_std / np.sqrt(len(test_acc_all)))
        if output_ari:
            return final_mean, final_std, final_interval, nmi_all, ari_all
        else:
            return final_mean, final_std, final_interval

    def train(self):
        bad_counter = 0
        id_support_val, id_query_val, class_selected_val = \
            tasks_generator(self.data.id_by_class, self.data.val_class, args.n_way, args.k_shot, args.m_qry, args.num_test)
        for epoch in range(args.num_epoch):
            encoder_loss = 0
            acc_vals = []

            ## train cls ##
            id_support, id_query, class_selected = \
                tasks_generator(self.data.id_by_class, self.data.train_class, args.n_way, args.k_shot, args.m_qry, args.num_train)
            self.se_net.train()
            self.opti_cls.zero_grad()
            cls_loss, acc_before = self.train_cls(id_support, id_query, class_selected)
            with torch.autograd.detect_anomaly():
                cls_loss.backward()
            self.opti_cls.step()

            for i in range(args.num_train):
                self.se_net.train()
                self.opti_encoder.zero_grad()
                self.opti_cls.zero_grad()
                y_support = torch.tensor(
                    [class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_support[i]]]).to(self.device)
                y_query = torch.tensor(
                    [class_selected[i].index(j) for j in torch.squeeze(self.data.y)[id_query[i]]]).to(self.device)
                mi_loss, cls_loss = self.train_encoder(id_support[i], id_query[i], y_support, y_query)
                #print(cls_loss)
                loss = mi_loss - args.beta * cls_loss
                encoder_loss += loss.data.cpu().numpy()

                with torch.autograd.detect_anomaly():
                    loss.backward()
                self.opti_encoder.step()
            self.scheduler.step()

            ## validation ##
            self.se_net.eval()
            for index, param_c in enumerate([0.1]):
                acc_val, _, _ = self.fs_test(id_support_val, id_query_val, class_selected_val, param_c,
                                             solver=args.solver, iter_clf=args.iter_clf)
                acc_vals.append(acc_val)
                if acc_val > self.best_acc_val:
                    print("better graph encoder!")
                    self.best_acc_val = acc_val
                    self.best_param_c = param_c
                    self.best_encoder_weight = deepcopy(self.se_net.graph_encoder.state_dict())
            if self.best_acc_val > self.old_best_acc:
                self.old_best_acc = self.best_acc_val
                bad_counter = 0
            else:
                bad_counter += 1

                if bad_counter > args.patience:
                    break

            print("EPOCH ", epoch, "\tCUR_LOSS_Train ", encoder_loss / args.num_train, "\tCUR_ACC_Val ", max(acc_vals),
                  "\tBEST_ACC_VAL ", self.best_acc_val)

        ## test ##
        with torch.no_grad():
            self.se_net.graph_encoder.load_state_dict(self.best_encoder_weight)
            self.se_net.eval()
            id_support, id_query, class_selected = \
                tasks_generator(self.data.id_by_class, self.data.test_class, args.n_way, args.k_shot, args.m_qry, args.num_test)
            acc_test, std_test, interval_test,nmi,ari = self.fs_test(id_support, id_query, class_selected, self.best_param_c,
                                                             args.solver, args.iter_clf,True)

            print("Test_Acc: ", acc_test, "\tTest_Std: ", std_test, "\tTest_Interval: ", interval_test)

            return acc_test,interval_test,nmi,ari


if __name__ == '__main__':
    args = set_params()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    start = time.time()
    train = TrainFlow()
    acc_test, interval_test, nmis, aris = train.train()
    end = time.time()
    result_file = open('./SE-FSNC-result-{}.txt'.format(args.dataset_name), 'a')
    result_file.write('{}-way {}-shot'.format(args.n_way, args.k_shot) + '\n'
                +'acc: {:.4f}\n'.format(acc_test)
                + 'interval: {:.4f} '.format(interval_test) + '\n'
                + 'nmi: {:.4f}'.format(nmis) + '\n'
                + 'ari: {:.4f}'.format(aris) + '\n\n')
    print(f'Finished in {(end - start) / 60:.1f} min')
