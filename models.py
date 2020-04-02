import torch
import torch.nn as nn
import torch.nn.functional as F

#VAE model
class VAE_encoder(nn.Module):
    def __init__(self, latent_dim, num_classes, device):
        super(VAE_encoder, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device

        self.down_sample = nn.Sequential(
            nn.Conv3d(1, 8, 5, stride=2), #Nx8x62x62x62
            nn.BatchNorm3d(8),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 5, stride=2), #Nx16x29x29x29
            nn.BatchNorm3d(16),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, 5, stride=2), #Nx32x13x13x13
            nn.BatchNorm3d(32),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2), #Nx64x6x6x6
            nn.BatchNorm3d(64),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True)
        )
        self.flatten = nn.Sequential(
            nn.Conv3d(64, 128, 6), #Nx128x1x1x1
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(), #Nx128
        )
        self.age_mean = nn.Linear(128+1, self.num_classes)
        self.age_log_var = nn.Linear(128+1, self.num_classes)
        self.z_mean = nn.Linear(128+1, self.latent_dim)
        self.z_log_var = nn.Linear(128+1, self.latent_dim)
        self.z_cond_age_mean = nn.Linear(self.num_classes, self.latent_dim)

    def sample(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(mu.shape, dtype=torch.float32).to(self.device)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, img, gender):
        x = self.down_sample(img)
        x = self.flatten(x)
        x = torch.cat([x, gender.view(-1, 1)], dim=1)
        a_mean = self.age_mean(x)
        a_log_var = self.age_log_var(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        a = self.sample(a_mean, a_log_var)
        z = self.sample(z_mean, z_log_var)
        z_a = self.z_cond_age_mean(a)
        return a_mean, a_log_var, z_mean, z_log_var, a, z, z_a
    
    
class UnFlatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        num_ch = x.shape[1]
        return x.view(batch_size, num_ch, 1, 1, 1)


class VAE_decoder(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_decoder, self).__init__()
        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose3d(self.latent_dim, 64, 5),
            nn.BatchNorm3d(64),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 5, stride=2),
            nn.BatchNorm3d(32),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, 5, stride=2),
            nn.BatchNorm3d(16),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, 5, stride=2),
            nn.BatchNorm3d(8),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(8, 1, 5, stride=2),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(1, 1, 4)
        )

    def forward(self, z):
        return self.decoder(z)
    

class VAE_age(nn.Module):
    def __init__(self, latent_dim, num_classes, device):
        super(VAE_age, self).__init__()
        
        self.encoder = VAE_encoder(latent_dim, num_classes, device)
        self.decoder = VAE_decoder(latent_dim)

    def forward(self, img, gender):
        (a_mean, a_log_var, z_mean, 
              z_log_var, a, z, z_a) = self.encoder(img, gender)
        img_rec = self.decoder(z)
        return (a_mean, a_log_var, z_mean, z_log_var, z_a,
                a, img_rec)
    
    
    
##CNN model
class S500MRI_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(S500MRI_block, self).__init__()
        hid_channels = out_channels // 2
        self.s500mri = nn.Sequential(
            nn.Conv3d(in_channels, hid_channels, 3),
            nn.BatchNorm3d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),
            nn.Conv3d(hid_channels, out_channels, 3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.s500mri(x)


class S500MRI_clf_block(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(S500MRI_clf_block, self).__init__()
        self.flatten = nn.Sequential(
            nn.Conv3d(in_dim, in_dim * 2, 4),
            nn.RReLU(inplace=True),
            nn.Flatten()
        )
        self.clf = nn.Linear(in_dim * 2 + 1, num_classes)

    def forward(self, img, gender):
        out = self.flatten(img)
        out = torch.cat([out, gender.view(-1, 1)], dim=1)
        return self.clf(out)
        

class S500MRI_clf(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=4):
        super(S500MRI_clf, self).__init__()
        layers = []
        in_dim = 1
        out_dim = 8
        for _ in range(4):
            layers.append(S500MRI_block(in_dim, out_dim))
            in_dim = out_dim
            out_dim = in_dim * 2

        self.features = nn.Sequential(*layers)
        self.clf = S500MRI_clf_block(in_dim, num_classes)
        
        
    def forward(self, img, gender):
        out = self.features(img)
        return self.clf(out, gender)