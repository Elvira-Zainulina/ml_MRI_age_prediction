import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

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
        

def training_plot(losses, losses_dev, f1_scores, 
                  f1_scores_dev, out_path=None):
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(losses, lw=3, label='train')
    ax1.plot(losses_dev, lw=3, label='validation')
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_xlabel('num_of_epoch', fontsize=14)
    ax2.plot(f1_scores, lw=3, label='train')
    ax2.plot(f1_scores_dev, lw=3, label='validation')
    ax2.set_ylabel('F1-score', fontsize=14)
    ax2.set_xlabel('num_of_epoch', fontsize=14)
    ax1.grid()
    ax2.grid()
    ax1.legend(fontsize=14, loc=1)
    ax2.legend(fontsize=14, loc=4)
    if out_path:
        plt.savefig(out_path)
    plt.show()
    

def train_epoch(dataloader, model, optimizer, loss_fn, 
                scheduler=None, device='cpu'):
    
    model.train()
    train_loss = 0
#     train_acc = 0
    train_preds = []
    train_labels = []
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
            
        optimizer.zero_grad()
        output = model(img, gender)
        loss = loss_fn(output, age)
        train_loss += loss.item()
        train_preds.append(output.argmax(1).cpu().numpy())
        train_labels.append(age.cpu().numpy())
#         f1 = f1_score(age.cpu(), o)
#         train_acc += (output.argmax(1) == age).sum().float() / output.shape[0]
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    train_preds = np.concatenate(train_preds)
    train_labels = np.concatenate(train_labels)
    return train_loss / (i + 1), f1_score(train_labels, train_preds, average='micro') #train_acc / (i + 1)


def test(dataloader, model, loss_fn, device='cpu'):
    model.eval()
    test_loss = 0
#     test_acc = 0
    test_preds = []
    test_labels = []
    for i, sample in tqdm(enumerate(dataloader)):
        img, gender, age = sample
        img = img.to(device)
        age = age.to(device)
        gender = gender.to(device)
        
        output = model(img, gender)
        test_loss += loss_fn(output, age).item()
#         test_acc += (output.argmax(1) == age).sum().float() / output.shape[0]
        test_preds.append(output.argmax(1).cpu().numpy())
        test_labels.append(age.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    return test_loss / (i + 1), f1_score(test_labels, test_preds, average='micro') #test_acc / (i + 1)


def train(train_dataloader, val_dataloader,
          model, optimizer, loss_fn, epochs,
          scheduler=None, device='cpu', 
          output_clf='clf.pt',
          draw=False, verbose=False):
          
    losses = []
    losses_dev = []
    f1_scores = []
    f1_scores_dev = []
    
    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(train_dataloader, model, optimizer,
                                           loss_fn=loss_fn, scheduler=scheduler,
                                           device=device)
        
        val_loss, val_f1 = test(val_dataloader, model, loss_fn=loss_fn,
                                device=device)
        
        losses.append(train_loss)
        losses_dev.append(val_loss)
        f1_scores.append(train_f1)
        f1_scores_dev.append(val_f1)             
        
        if draw:
            
            clear_output(wait=True)
            training_plot(losses, losses_dev, f1_scores, f1_scores_dev)
        
        if verbose:
            print(('Epoch {}: train_loss {:.4f}, val_loss {:.4f} \t'+\
                   'train_f1_score {:.4f}, val_f1_score {:.4f} ').format(
                    epoch, losses[-1], losses_dev[-1],
                    f1_scores[-1], f1_scores_dev[-1]))
            
        if (epoch + 1) % 10 == 0:
            torch.save(model, output_clf)
    
    print('Finished training.')
    return losses, losses_dev, f1_scores, f1_scores_dev


def train_epoch_VAE(dataloader, model, optimizer,
                    mse_loss, cross_entropy,
                    scheduler=None, device='cpu'):
    
    model.train()
    train_loss = 0
#     train_acc = 0
    train_preds = []
    train_labels = []
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
        train_preds.append(a_mean.argmax(1).cpu().numpy())
        train_labels.append(age.cpu().numpy())
#         train_acc += (a_mean.argmax(1) == age).sum().float() / age.shape[0]
        vae_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    train_preds = np.concatenate(train_preds)
    train_labels = np.concatenate(train_labels)

    return train_loss / (i + 1), f1_score(train_labels, train_preds, average='micro') #train_acc / (i + 1)


def test_VAE(dataloader, model, mse_loss, cross_entropy, device='cpu'):
    model.eval()
    test_loss = 0
#     test_acc = 0
    test_preds = []
    test_labels = []
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
#         test_acc += (a_mean.argmax(1) == age).sum().float() / age.shape[0]
        test_preds.append(a_mean.argmax(1).cpu().numpy())
        test_labels.append(age.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    return test_loss / (i + 1), f1_score(test_labels, test_preds, average='micro') #test_acc / (i + 1)


def train_VAE(train_dataloader, val_dataloader, model, 
              optimizer, mse_loss, cross_entropy, epochs,
              scheduler=None, device='cpu', output_clf='clf_VAE.pt',
              draw=False, verbose=False):
          
    losses = []
    losses_dev = []
    f1_scores = []
    f1_scores_dev = []
    
    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch_VAE(train_dataloader, model, optimizer,
                                               mse_loss, cross_entropy,
                                               scheduler=scheduler, device=device)
        
        val_loss, val_f1 = test_VAE(val_dataloader, model, mse_loss, 
                                    cross_entropy, device=device)
        
        losses.append(train_loss)
        losses_dev.append(val_loss)
        f1_scores.append(train_f1)
        f1_scores_dev.append(val_f1)             
        
        if draw:
            
            clear_output(wait=True)
            training_plot(losses, losses_dev, f1_scores, f1_scores_dev)
        
        if verbose:
            print(('Epoch {}: train_loss {:.4f}, val_loss {:.4f} \t'+\
                   'train_f1_score {:.4f}, val_f1_score {:.4f} ').format(
                    epoch, losses[-1], losses_dev[-1],
                    f1_scores[-1], f1_scores_dev[-1]))
            
        if (epoch + 1) % 10 == 0:
            torch.save(model, output_clf)
    
    print('Finished training.')
    return losses, losses_dev, f1_scores, f1_scores_dev