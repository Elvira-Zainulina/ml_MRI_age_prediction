import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm

def set_random_seeds(seed_value=0, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def train_epoch(dataloader, model, optimizer, loss_fn, 
                scheduler=None, device='cpu'):
    
    model.train()
    train_loss = 0
    train_acc = 0
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
            
        optimizer.zero_grad()
        output = model(img, gender)
        loss = loss_fn(output, age)
        train_loss += loss.item()
        train_acc += (output.argmax(1) == age).sum().float() / output.shape[0]
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return train_loss / (i + 1), train_acc / (i + 1)


def test(dataloader, model, loss_fn, device='cpu'):
    model.eval()
    test_loss = 0
    test_acc = 0
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
        
        output = model(img, gender)
        test_loss += loss_fn(output, age).item()
        test_acc += (output.argmax(1) == age).sum().float() / output.shape[0]

    return test_loss / (i + 1), test_acc / (i + 1)


def train(train_dataloader, val_dataloader,
          model, optimizer, loss_fn, epochs,
          scheduler=None, device='cpu', 
          output_clf='clf.pt',
          draw=False, verbose=False):
          
    losses = []
    losses_dev = []
    accuracies = []
    accuracies_dev = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(train_dataloader, model, optimizer,
                                            loss_fn=loss_fn, scheduler=scheduler,
                                            device=device)
        
        val_loss, val_acc = test(val_dataloader, model, loss_fn=loss_fn,
                                 device=device)
        
        losses.append(train_loss)
        losses_dev.append(val_loss)
        accuracies.append(train_acc)
        accuracies_dev.append(val_acc)             
        
        if draw:
            
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.plot(losses, lw=3, label='train')
            ax1.plot(losses_dev, lw=3, label='dev')
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.set_xlabel('num_of_epoch', fontsize=14)
            ax2.plot(accuracies, lw=3, label='train')
            ax2.plot(accuracies_dev, lw=3, label='dev')
            ax2.set_ylabel('Accuracy', fontsize=14)
            ax2.set_xlabel('num_of_epoch', fontsize=14)
            ax1.legend(fontsize=14, loc=1)
            ax2.legend(fontsize=14, loc=4)
            plt.show()
        
        if verbose:
            print(('Epoch {}: train_loss {:.4f}, val_loss {:.4f} \t'+\
                   'train_accuracy {:.4f}, val_accuracy {:.4f} ').format(
                    epoch, losses[-1], losses_dev[-1],
                    accuracies[-1], accuracies_dev[-1]))
            
        if (epoch + 1) % 10 == 0:
            torch.save(model, output_clf)
    
    print('Finished training.')
    return losses, losses_dev, accuracies, accuracies_dev


def train_epoch_VAE(dataloader, model, optimizer,
                    mse_loss, cross_entropy,
                    scheduler=None, device='cpu'):
    
    model.train()
    train_loss = 0
    train_acc = 0
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
            
        optimizer.zero_grad()
        a_mean, a_log_var, z_mean, z_log_var, z_a, a, img_rec = model(img, gender)

        reconstr_loss = mse_loss(img, img_rec)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + 
                                             (z_mean - z_a) ** 2 - 1. - z_log_var, 1))
        label_loss = cross_entropy(a_mean, age)
        vae_loss = reconstr_loss + kl_loss + label_loss

        train_loss += (vae_loss).item()
        train_acc += (a_mean.argmax(1) == age).sum().float() / age.shape[0]
        vae_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return train_loss / (i + 1), train_acc / (i + 1)


def test_VAE(dataloader, model, mse_loss, cross_entropy, device='cpu'):
    model.eval()
    test_loss = 0
    test_acc = 0
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
        
        a_mean, a_log_var, z_mean, z_log_var, z_a, a, img_rec = model(img, gender)

        reconstr_loss = mse_loss(img, img_rec)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_log_var) + 
                                             (z_mean - z_a) ** 2 - 1. - z_log_var, 1))
        label_loss = cross_entropy(a_mean, age)
        vae_loss = reconstr_loss + kl_loss + label_loss

        test_loss += vae_loss.item()
        test_acc += (a_mean.argmax(1) == age).sum().float() / age.shape[0]

    return test_loss / (i + 1), test_acc / (i + 1)


def train_VAE(train_dataloader, val_dataloader, model, 
              optimizer, mse_loss, cross_entropy, epochs,
              scheduler=None, device='cpu', output_clf='clf_VAE.pt',
              draw=False, verbose=False):
          
    losses = []
    losses_dev = []
    accuracies = []
    accuracies_dev = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_VAE(train_dataloader, model, optimizer,
                                                mse_loss, cross_entropy,
                                                scheduler=scheduler, device=device)
        
        val_loss, val_acc = test_VAE(val_dataloader, model, mse_loss, 
                                     cross_entropy, device=device)
        
        losses.append(train_loss)
        losses_dev.append(val_loss)
        accuracies.append(train_acc)
        accuracies_dev.append(val_acc)             
        
        if draw:
            
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.plot(losses, lw=3, label='train')
            ax1.plot(losses_dev, lw=3, label='dev')
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.set_xlabel('num_of_epoch', fontsize=14)
            ax2.plot(accuracies, lw=3, label='train')
            ax2.plot(accuracies_dev, lw=3, label='dev')
            ax2.set_ylabel('Accuracy', fontsize=14)
            ax2.set_xlabel('num_of_epoch', fontsize=14)
            ax1.legend(fontsize=14, loc=1)
            ax2.legend(fontsize=14, loc=4)
            plt.show()
        
        if verbose:
            print(('Epoch {}: train_loss {:.4f}, val_loss {:.4f} \t'+\
                   'train_accuracy {:.4f}, val_accuracy {:.4f} ').format(
                    epoch, losses[-1], losses_dev[-1],
                    accuracies[-1], accuracies_dev[-1]))
            
        if (epoch + 1) % 10 == 0:
            torch.save(model, output_clf)
    
    print('Finished training.')
    return losses, losses_dev, accuracies, accuracies_dev