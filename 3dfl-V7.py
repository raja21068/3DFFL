import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from abc import abstractmethod
import copy

# Define PointNet++ for feature extraction (simplified version)
class PointNetPP(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(PointNetPP, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.final_feat_dim = feature_dim

    def forward(self, x):
       # x = x.transpose(1, 2)  # Transpose to (B, D, N)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, 2)[0]  # Max pooling
        x = self.fc(x)
        return x

# Define the attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights
        return x


def calculate_embedding_loss(predicted_scores, ground_truth_labels):
    # Apply softmax to get probabilities from predicted scores
    predicted_probabilities = F.softmax(predicted_scores, dim=1)

    # Convert ground truth labels to one-hot encoding if they aren't already
    if ground_truth_labels.dim() == 2:  # Check if already one-hot encoded
        num_classes = predicted_probabilities.size(1)
        ground_truth_labels = F.one_hot(ground_truth_labels.argmax(dim=1), num_classes=num_classes).float()

    # Calculate the dot product between predicted probabilities and ground truth labels
    loss = -torch.mean(torch.sum(ground_truth_labels * predicted_probabilities, dim=1))
    return loss

def calculate_learnable_loss(local_prototypes, local_queries, ground_truth_labels):
    distances = torch.cdist(local_queries, local_prototypes, p=2)  # Compute L2 distances
    exp_distances = torch.exp(-distances)
    probabilities = exp_distances / exp_distances.sum(dim=1, keepdim=True)

    ground_truth_labels = ground_truth_labels.argmax(dim=1)  # Convert one-hot to class indices
    ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=probabilities.size(1)).float()

    loss = -torch.mean(torch.sum(ground_truth_labels * torch.log(probabilities + 1e-10), dim=1))
    return loss

def calculate_overall_loss(embedding_loss, learnable_loss, comp_loss, lambda1, lambda2, lambda3):
    overall_loss = lambda1 * embedding_loss + lambda2 * learnable_loss + lambda3 * comp_loss
    return overall_loss

def differential_privacy_update(model, noise_multiplier=0.1):
    for param in model.parameters():
        noise = torch.normal(0, noise_multiplier, size=param.size()).to(param.device)
        param.data.add_(noise)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def select_diverse_clients(clients_data, num_clients):
    selected_clients = np.random.choice(clients_data, num_clients, replace=False)
    return selected_clients

def stratified_partitioning(data, labels, num_clients):
    from sklearn.model_selection import train_test_split
    client_data = []
    client_labels = []
    for _ in range(num_clients):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=1/num_clients)
        client_data.append(X_train)
        client_labels.append(y_train)
    return list(zip(client_data, client_labels))

def federated_training_process(num_rounds, local_datasets, global_model, lambda1, lambda2, lambda3, apply_privacy=False):
    num_nodes = len(local_datasets)
    momentum = 0.9
    global_momentum = {key: torch.zeros_like(value).float() for key, value in global_model.state_dict().items()}
    previous_loss = float('inf')

    for t in range(num_rounds):
        local_updates = []
        total_loss = 0

        for i in range(num_nodes):
            X_i, Y_i = local_datasets[i]
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)

            local_model.train()
            for epoch in range(5):
                optimizer.zero_grad()
                features = local_model.feature(X_i)

                attention_layer = AttentionLayer(features.size(1)).to(features.device)
                attended_features = attention_layer(features)

                embedding_loss = calculate_embedding_loss(attended_features, Y_i)
                learnable_loss = calculate_learnable_loss(attended_features, features, Y_i)
                comp_loss = torch.tensor(0.1).to(features.device)
                overall_loss = calculate_overall_loss(embedding_loss, learnable_loss, comp_loss, lambda1, lambda2, lambda3)
                overall_loss.backward()
                optimizer.step()

                total_loss += overall_loss.item()

            if apply_privacy:
                differential_privacy_update(local_model)

            local_updates.append(local_model.state_dict())

        average_loss = total_loss / num_nodes
        if abs(previous_loss - average_loss) < 1e-3:  # Convergence criterion
            break
        previous_loss = average_loss

        global_model_dict = global_model.state_dict()
        for key in global_model_dict.keys():
            updates = torch.stack([local_updates[i][key].float() for i in range(num_nodes)], dim=0)
            average_update = updates.mean(dim=0)
            global_momentum[key] = momentum * global_momentum[key] + (1 - momentum) * average_update
            global_model_dict[key] = global_model_dict[key].float() + global_momentum[key]

        global_model.load_state_dict(global_model_dict)

        for i in range(num_nodes):
            local_datasets[i][0].model = copy.deepcopy(global_model)

    return global_model

# MetaTemplate class
class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=False):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = 15  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            if x.size()[0] != self.n_way * (self.n_support + self.n_query):
                x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            x = x.permute(0, 2, 1)
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d} | Loss {:f}'.format(epoch, i, avg_loss / float(i + 1)))
        return avg_loss / float(i + 1)

    def test_loop(self, test_loader, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = 100
        for i, (x, _) in enumerate(test_loader):
            self.n_query = 15
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature=True):  # further adaptation, default is fixing feature and training a new softmax classifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)
        return F.cross_entropy(scores, y_query)

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

# Example usage
class GlobalModel(ProtoNet):
    def __init__(self, input_dim, feature_dim, num_classes):
        super(GlobalModel, self).__init__(lambda: PointNetPP(input_dim, feature_dim), num_classes, 5)

def load_data():
    N = 100
    P = 1024
    D = 3
    C = 10
    return [(torch.rand(N, D, P), torch.randint(0, 2, (N, C)).float()) for _ in range(5)]

def create_few_shot_batches(data, n_way, k_shot, q_query):
    support_set = []
    query_set = []
    labels = []

    for i in range(n_way):
        class_samples = data[i]
        support_samples = class_samples[:k_shot]
        query_samples = class_samples[k_shot:k_shot + q_query]

        support_set.append(support_samples)
        query_set.append(query_samples)
        labels.append(i)

    support_set = torch.cat(support_set, 0)
    query_set = torch.cat(query_set, 0)
    labels = torch.tensor(labels)

    return support_set, query_set, labels

def few_shot_training(global_model, data, num_rounds, n_way, k_shot, q_query):
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for round in range(num_rounds):
        global_model.train()
        optimizer.zero_grad()

        support_set, query_set, labels = create_few_shot_batches(data, n_way, k_shot, q_query)
        support_features = global_model.feature(support_set)
        query_features = global_model.feature(query_set)

        prototypes = support_features.view(n_way, k_shot, -1).mean(dim=1)

        dists = torch.cdist(query_features, prototypes)
        logits = -dists

        # Reshape logits to match the label dimensions
        logits = logits.view(-1, n_way)
        labels = labels.repeat_interleave(q_query) # Adjust labels for each query sample

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        print(f'Round {round + 1}/{num_rounds}, Loss: {loss.item()}')

    return global_model

# Main execution
local_datasets = load_data()
global_model = GlobalModel(input_dim=3, feature_dim=128, num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')
lambda1, lambda2, lambda3 = 1.0, 1.0, 0.1

# Federated training
trained_global_model = federated_training_process(num_rounds=50, local_datasets=local_datasets, global_model=global_model, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, apply_privacy=True)

# Few-shot learning
n_way, k_shot, q_query = 5, 5, 15
few_shot_data = [torch.rand((100, 3, 1024)) for _ in range(n_way)]  # Example few-shot data
trained_few_shot_model = few_shot_training(trained_global_model, few_shot_data, num_rounds=10, n_way=n_way, k_shot=k_shot, q_query=q_query)

# Save the trained model state
torch.save(trained_few_shot_model.state_dict(), 'few_shot_trained_model.pth')
