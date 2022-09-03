# from torch.utils.tensorboard import SummaryWriter

import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from Model import GlobalLocal


# writer = SummaryWriter()


def show_curve(ys, title):
    """plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: Loss or Accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()


def show_metric(ys, title):
    """plot curlve for HR and NDCG
    Args:
        ys: hr or ndcg list
        title: HR@k or NDCG@k
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('User')
    plt.ylabel('{} Value'.format(title))
    plt.show()


class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count):
        super(ConvNCF, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.embedding_size = 64
        self.P = nn.Embedding(self.user_count, self.embedding_size).cuda()
        self.Q = nn.Embedding(self.item_count, self.embedding_size).cuda()

        # cnn setting
        self.channel_size = 32
        self.cnn = GlobalLocal(1, self.channel_size)
        self.fc = nn.Linear(32, 1)

    def forward(self, user_ids, item_ids):
        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).cuda())
        item_embeddings = self.Q(torch.tensor(item_ids).cuda())

        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
        interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

        # cnn
        feature_map = self.cnn(interaction_map)  # output: batch_size * 32 * 1 * 1
        feature_vec = feature_map.view((-1, 32))

        # fc
        prediction = self.fc(feature_vec)
        prediction = prediction.view((-1))
        return prediction


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        #         print('loss:', loss)
        return loss

def train():
    lr = 0.001
    epoches = 20
    batch_size = 256
    losses = []
    accuracies = []

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(yelp.train_group, batch_size=batch_size, shuffle=True, num_workers=4)

    bpr_loss = BPRLoss().cuda()

    for epoch in range(epoches):

        total_loss = 0
        total_acc = 0
        try:
            with tqdm(train_loader) as t:
                for batch_idx, train_data in enumerate(t):
                    #             train_data = Variable(train_data.cuda())
                    user_ids = Variable(train_data[:, 0].cuda())
                    pos_item_ids = Variable(train_data[:, 1].cuda())
                    neg_item_ids = Variable(train_data[:, 2].cuda())

                    optimizer.zero_grad()

                    # train convncf
                    pos_preds = model(user_ids, pos_item_ids)
                    neg_preds = model(user_ids, neg_item_ids)

                    loss = bpr_loss(pos_preds, neg_preds)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    accuracy = float(((pos_preds - neg_preds) > 0).sum()) / float(len(pos_preds))
                    total_acc += accuracy

        except KeyboardInterrupt:
            t.close()
            raise

        losses.append(total_loss / (batch_idx + 1))
        accuracies.append(total_acc / (batch_idx + 1))
        print('Epoch:', epoch, 'Train loss:', losses[-1], 'Train acc:', accuracies[-1])
        if epoch % 2 == 0 and epoch != 0:
            # show_curve(losses, 'train loss')
            # show_curve(accuracies, 'train acc')
            evaluate()


best_hr = 0.0
best_ndcg = 0.0


def evaluate():
    model.eval()
    hr5 = []
    hr10 = []
    hr20 = []
    ndcg5 = []
    ndcg10 = []
    ndcg20 = []
    global best_hr
    global best_ndcg

    user_count = len(yelp.test_negative)
    try:
        with tqdm(range(user_count)) as t:
            for u in t:
                item_ids = torch.tensor(yelp.test_negative[u]).cuda()
                user_ids = torch.tensor([u] * len(item_ids)).cuda()
                predictions = model(user_ids, item_ids)
                topv, topi = torch.topk(predictions, 10, dim=0)
                # hr, ndcg = scoreK(topi, 5)
                # hr5.append(hr)
                # ndcg5.append(ndcg)
                hr, ndcg = scoreK(topi, 10)
                hr10.append(hr)
                ndcg10.append(ndcg)
                # hr, ndcg = scoreK(topi, 20)
                # hr20.append(hr)
                # ndcg20.append(ndcg)

            print('HR@10:', sum(hr10) / len(hr10))
            with open('hit.txt', 'a') as f:
                print("{:.4f}".format(sum(hr10) / len(hr10)), file=f)

            print('NDCG@10:', sum(ndcg10) / len(ndcg10))
            with open('ndcg.txt', 'a') as f:
                print("{:.4f}".format(sum(ndcg10) / len(ndcg10)), file=f)

            # print('HR@20:', sum(hr20) / len(hr20))
            # print('NDCG@20:', sum(ndcg20) / len(ndcg20))
            if (sum(hr10) / len(hr10)) > best_hr:
                best_hr = sum(hr10) / len(hr10)

            if (sum(ndcg10) / len(ndcg10)) > best_ndcg:
                best_ndcg = (sum(ndcg10) / len(ndcg10))
    except KeyboardInterrupt:
        t.close()
        raise


def scoreK(topi, k):
    hr = 1.0 if 99 in topi[0:k] else 0.0
    if hr:
        ndcg = math.log(2) / math.log(topi.tolist().index(99) + 2)
    else:
        ndcg = 0
    # auc = 1 - (position * 1. / negs)
    return hr, ndcg  # , auc


if __name__ == '__main__':
    # torch.set_num_threads(12)
    print('Data loading...')
    yelp = load_dataset.Load_Yelp('./Data/pinterest-20.train.rating', './Data/pinterest-20.test.rating',
                                  './Data/pinterest-20.test.negative')
    print('Data loaded')

    print('=' * 50)

    print('Model initializing...')
    model = ConvNCF(int(max(yelp.train_group[:, 0])) + 1, int(max(yelp.train_group[:, 1])) + 1).cuda()

    print('Model initialized')

    print('=' * 50)

    print('Model training...')
    train()
    print('Model trained')

    print('=' * 50)

    print('Model evaluating...')
    evaluate()
    print("best HR ", best_hr)
    print("best NDcg", best_ndcg)
    print('Model evaluated')
