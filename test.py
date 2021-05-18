import numpy as np
import torch 

def test(meta, model, loader, args):
    model.eval()
    with torch.no_grad():
        correct = []
        for batch in loader:
            if meta:
                m, x_ = batch
                m = m.to(args.device) # [batch, n_train, sample (with y)]
                x = x_[:,:,:-1].to(args.device) # [batch, n_test, input (no y)]
                y = x_[:,:,-1].type(torch.long).to(args.device)
                y = y.squeeze() # y: [batch, n_test]
                y_hat, attention = model(x, m) # yhat: [batch, n_test, 2]
                preds = torch.argmax(y_hat, dim=2).squeeze(0) # [n_test]
            else:
                f1, f2, ax, y = batch # face1, face2, axis, y
                f1 = f1.to(args.device)
                f2 = f2.to(args.device)
                ax = ax.to(args.device)
                y = y.to(args.device).squeeze()
                y_hat = model(f1, f2, ax) # [batch, 2]
                preds = torch.argmax(y_hat, dim=1).squeeze(0) # [batch]
            c = (preds == y)
            c = c.cpu().numpy().tolist()
            correct += c
    acc = np.mean(correct)
    return acc

