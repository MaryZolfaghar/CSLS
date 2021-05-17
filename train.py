import numpy as np
import torch
import torch.nn as nn


def train(meta, model, loader, args):
    # model
    model.train()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    lr = args.lr_episodic if meta else args.lr_cortical
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    train_losses = [] # for recording all train losses
    ave_loss = [] # running average loss for printing
    N = args.N_episodic if meta else args.N_cortical 
    i = 0
    done = False
    while not done:
        for batch in loader:
            optimizer.zero_grad()
            if meta:
                m, x_ = batch
                m = m.to(args.device) # [batch, n_train, sample (with y)]
                x = x_[:,:,:-1].to(args.device) # [batch, n_test, input (no y)]
                y = x_[:,:,-1].type(torch.long).to(args.device) 
                # y: [batch, n_test, 1]
                y_hat, attention = model(x, m) # yhat: [batch, n_test, 2]
                y_hat = y_hat.view(-1, y_hat.shape[2]) # [batch*n_test, 2]
                y = y.view(-1) # [batch*n_test]
            else:
                f1, f2, ax, y = batch # face1, face2, axis, y
                f1 = f1.to(args.device)
                f2 = f2.to(args.device)
                ax = ax.to(args.device)
                y = y.to(args.device).squeeze(1)
                y_hat = model(f1, f2, ax)
            # Loss
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            # Record loss
            train_losses.append(loss.data.item())
            ave_loss.append(loss.data.item())

            if i % args.print_every == 0:
                print("Step: {}, Loss: {}".format(i, np.mean(ave_loss)))
                ave_loss = []
            if i >= N:
                done = True 
                break
            i += 1

    return train_losses


    