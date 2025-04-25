import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def vis(pred, truths):
    for p, tru in zip(pred, truths):
        print(sum(tru))
        plt.figure(figsize=(12,4))
        plt.plot(np.arange(0, len(p), 1) , np.maximum(0, p))
        for j, t in enumerate(tru):
            if t != 0:
                #plt.axvline(j, color='red', linestyle='-')
                pass
        plt.show()

        input()



def evaluate(model, test_loader, loss_fn, device, visualize = False):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for data in test_loader:
            data, labels = data
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            if visualize:
                vis(output.cpu(), labels.cpu())

            loss = loss_fn(output, labels)

            test_loss += loss 
    print(f'Test loss: {test_loss / len(test_loader):.8f}')

    model.train()

    return test_loss


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, save_model=False, model_name="", scheduler=None, device = "cpu"):
    model.train()
    best_loss = evaluate(model, test_loader, loss_fn, device=device)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}: {epoch_loss / len(train_loader):.8f}')

        if (epoch + 1) % 5 == 0:
            eval_loss = evaluate(model, test_loader, loss_fn, device=device)
            if save_model and eval_loss<best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), os.path.join('trained_models', model_name + f'_{epoch + 1}.pt'))

        if scheduler:
            scheduler.step()
    evaluate(model, test_loader, loss_fn, device=device, visualize = True)

    pass