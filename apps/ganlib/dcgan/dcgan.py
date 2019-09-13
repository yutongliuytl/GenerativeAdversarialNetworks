import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image,make_grid

import boto3
import s3fs
import pickle
from PIL import Image
import io

s3 = boto3.resource("s3")
client_s3 = boto3.client('s3')
client_db = boto3.client('dynamodb')


__all__ = [
    'DCGAN',
]

class Generator(nn.Module):
    
    def __init__(self, g_input_dim, g_output_dim, hidden_size=256):
        
        super(Generator, self).__init__()      
        
        self.layer = nn.Sequential(
                        nn.Linear(g_input_dim, hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_size, hidden_size*2),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_size*2, hidden_size*4),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_size*4, g_output_dim))
        
        self.output = nn.Tanh()
    
    # forward method
    def forward(self, x): 
        x = self.layer(x)
        return self.output(x)


class Discriminator(nn.Module):
            
    def __init__(self, d_input_dim, hidden_size=1024):
            
        super(Discriminator, self).__init__()
            
        self.layer = nn.Sequential(
                        nn.Linear(d_input_dim, hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),       
                        nn.Linear(hidden_size, hidden_size//2),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),       
                        nn.Linear(hidden_size//2, hidden_size//4),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_size//4, 1))
            
        self.output = nn.Sigmoid()
    
    # forward method
    def forward(self, x):
        x = self.layer(x)
        return self.output(x)


class DCGAN():

    def __init__(self,name,z_dim,dataset,lr_g = 0.0002,lr_d = 0.0002):
        
        self.name = name
        self.z_dim = z_dim
        self.data_dim = dataset.data.size(1) * dataset.data.size(2)

        self.G = Generator(g_input_dim = self.z_dim, g_output_dim = self.data_dim)
        self.D = Discriminator(self.data_dim)
        
        self.loss = nn.BCELoss() 
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = lr_g)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr = lr_d)


    #Hidden functions
    def _G_train(self,x,batch_size):
    
        self.G.zero_grad()

        z = Variable(torch.randn(batch_size, self.z_dim))
        y = Variable(torch.ones(batch_size, 1))

        G_output = self.G(z)
        D_output = self.D(G_output)
        G_loss = self.loss(D_output, y)

        G_loss.backward()
        self.G_optimizer.step()

        return G_loss.data.item()

    
    def _D_train(self,x,batch_size):
    
        self.D.zero_grad()

        x_real, y_real = x.view(-1, self.data_dim), torch.ones(batch_size, 1)
        x_real, y_real = Variable(x_real), Variable(y_real)

        D_output = self.D(x_real)
        D_real_loss = self.loss(D_output, y_real)
        D_real_score = D_output

        z = Variable(torch.randn(batch_size, self.z_dim))
        x_fake, y_fake = self.G(z), Variable(torch.zeros(batch_size, 1))

        D_output = self.D(x_fake)
        D_fake_loss = self.loss(D_output, y_fake)
        D_fake_score = D_output

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        self.D_optimizer.step()
            
        return  D_loss.data.item()


    def _save_image(self, tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):

        grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                        normalize=normalize, range=range, scale_each=scale_each)
        
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        in_mem_file = io.BytesIO()
        im.save(in_mem_file, format="JPEG")
        client_s3.put_object(Bucket="gan-dashboard",Key="generated-images/{0}/{1}.jpeg".format(self.name,filename),Body=in_mem_file.getvalue(),ACL='public-read')
        

    #Callable functions
    def fit(self,dataloader,max_epoch,batch_size):

        D_total_losses,G_total_losses = [],[]

        for epoch in range(1, max_epoch+1):           
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in enumerate(dataloader):
                D_losses.append(self._D_train(x,batch_size))
                G_losses.append(self._G_train(x,batch_size))
            D_total_losses.append(torch.mean(torch.FloatTensor(D_losses)))
            G_total_losses.append(torch.mean(torch.FloatTensor(G_losses)))
            self.generate(batch_size,epoch)
            
            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                    (epoch), max_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        return D_total_losses,G_total_losses


    def generate(self,batch_size,epoch=None,save=True):

        with torch.no_grad():
            test_z = Variable(torch.randn(batch_size, self.z_dim))
            generated = self.G(test_z)

            image = generated.view(generated.size(0), 1, 28, 28)
            
            if save:
                self._save_image(image, str(epoch))


    def load_state_dict(self,name_of_file='default_model'):

        # Loading discriminator
        data = client_s3.get_object(Bucket="gan-dashboard", Key="models/discriminator/{0}.pth".format(name_of_file))
        param = pickle.loads(data["Body"].read())
        self.D.load_state_dict(param)

        # Loading generator
        data = client_s3.get_object(Bucket="gan-dashboard", Key="models/generator/{0}.pth".format(name_of_file))
        param = pickle.loads(data["Body"].read())
        self.G.load_state_dict(param)

    
    def save_model(self,name_of_file='default_model'):

        # Saving discriminator
        data = pickle.dumps(self.D.state_dict()) 
        client_s3.put_object(Bucket="gan-dashboard",Key="models/discriminator/{0}.pth".format(name_of_file),Body=data)

        # Saving generator
        data = pickle.dumps(self.G.state_dict()) 
        client_s3.put_object(Bucket="gan-dashboard",Key="models/generator/{0}.pth".format(name_of_file),Body=data)