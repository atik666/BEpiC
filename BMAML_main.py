import torch
import numpy as np
from meta import Meta
from miniImageNet import MiniImagenet

import warnings
warnings.filterwarnings("ignore")

from    torch.utils.data import DataLoader
from tqdm import tqdm


n_way = 2
epochs = 10
k_shot = 5
k_query = 5


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)


    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda:0')
    maml = Meta(config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    train_expr = 'regular+bml'
    test_expr = 'regular'
    
    path = '/mnt/d/OneDrive - Oklahoma A and M System/RA/Fall 23/BMAML/Code/miniImageNet/'
    mini_train = MiniImagenet(path, mode='train', n_way=2, k_shot=k_shot,
                        k_query=k_query, batchsz=10000, resize=84, expr=train_expr)
    mini_test = MiniImagenet(path, mode='test', n_way=2, k_shot=k_shot,
                             k_query=k_query, batchsz=100, resize=84,  expr=test_expr)

    best_acc = 0.0
    for epoch in tqdm(range(epochs)):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini_train, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)
            
            
            if step % 100 == 0:
                print('\n','step:', step, '\ttraining acc:', accs)

            if step % 1000 == 0 or step == 2400:  # evaluation
                db_test = DataLoader(mini_test, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                if accs[-1] > best_acc:
                    best_acc = accs[-1]
                print('Best acc:', best_acc)
    print("\n"+"Train Method:",train_expr, "Test method:", test_expr,
          "Best test accuracy: ", best_acc)

main()    