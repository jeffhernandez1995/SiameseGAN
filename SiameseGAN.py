import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import datasets, transforms

import utils
from losses import ContrastiveLoss
from model import discriminator, generator


class SiameseGAN(object):
    """docstring for SiameseGAN"""
    def __init__(self, args):
        super(SiameseGAN, self).__init__()
        self.epoch = args.epoch
        self.sample_num = args.batch_size
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 100
        self.c = 0.01                   # clipping value
        self.n_critic = 1               # the number of iterations of the critic per generator iteration
        self.margin = 2.0
        self.margin_decay = 0.5

        self.transform = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=self.transform),
            batch_size=self.batch_size, shuffle=True) # Download Mnist

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        self.G = generator(64)
        self.G.apply(weights_init)
        self.D = discriminator(64)
        self.D.apply(weights_init)
        
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        if self.gpu_mode:
            self.G.to('cuda:0')
            self.D.to('cuda:0')
        self.CLloss = ContrastiveLoss()
        
        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim)).view(-1, 100, 1, 1)

        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.to('cuda:0')
        
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        print('training start!')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for i, (img, _) in enumerate(self.data_loader):
                #print(img.size())
                for p in self.D.parameters(): # reset requires_grad
                    p.requires_grad = True
                
                half_batch = img.size(0) // 2
                label_fake = torch.ones(half_batch)
                label_real = torch.zeros(half_batch)
                noise = torch.randn((half_batch,  self.z_dim)).view(-1, 100, 1, 1)
                if self.gpu_mode:
                    img, noise = img.to('cuda:0'), noise.to('cuda:0')
                    label_fake, label_real = label_fake.to('cuda:0'), label_real.to('cuda:0')
                #print(img.size())
                #print(torch.min(img), torch.max(img))
                self.D_optimizer.zero_grad()
                gen_img = self.G(noise.detach())
                output1, output2 = self.D(img[:half_batch,:,:,:], gen_img)
                d_loss_fake = self.CLloss(output1, output2, self.margin, label_fake)
                output1, output2 = self.D(img[:half_batch,:,:,:], img[half_batch:,:,:,:])
                d_loss_real = self.CLloss(output1, output2, self.margin, label_real)
                d_loss = d_loss_fake + d_loss_real
                d_loss.backward()
                self.D_optimizer.step()
                for p in self.D.parameters():
                    p.data.clamp_(-self.c, self.c)
                
                if ((i+1) % self.n_critic) == 0:
                    for p in self.D.parameters():
                        p.requires_grad = False
                    self.G_optimizer.zero_grad()

                    g_label_real = torch.zeros(img.size(0))
                    
                    g_noise = torch.randn((img.size(0), 100)).view(-1, 100, 1, 1)
                    if self.gpu_mode:
                        g_label_real, g_noise = g_label_real.to('cuda:0'), g_noise.to('cuda:0')
                    self.G_optimizer.zero_grad()
                    gen_img = self.G(g_noise)
                    #print(gen_img.size())
                    #print(img.size())
                    output1, output2 = self.D(img, gen_img)
                    g_loss = self.CLloss(output1, output2, self.margin, g_label_real)

                    self.train_hist['G_loss'].append(g_loss.item())
                    g_loss.backward()
                    self.G_optimizer.step()
                    self.train_hist['D_loss'].append(d_loss.item())
                if ((i + 1) % 100 )== 0:
                     print("Epoch: {} {} de {} D_loss: {:.8f} G_loss: {:.8f}".format(
                         (epoch + 1), (i + 1), self.data_loader.dataset.__len__() // self.batch_size,
                         d_loss.item(), g_loss.item()))

            self.margin = self.margin * self.margin_decay
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))
        print("Avg one epoch time: {:.2f}, total {} epochs time {:.2f}".format(np.mean(self.train_hist['per_epoch_time']),
           self.epoch, self.train_hist['total_time']))
        print("Training finish!... save training results")
        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        
    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim)).view(-1, 100, 1, 1)
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.mul(0.5).add(0.5)
            samples_cpu = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.mul(0.5).add(0.5)
            samples_cpu = samples.data.numpy().transpose(0, 2, 3, 1)

        #samples = (samples + 1) / 2
        #np.save('./results/mnist/vector_{}.npy'.format(epoch), samples)
        vutils.save_image(samples.detach(), self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '_vis.png')
        #utils.save_images(samples_cpu[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
#                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
