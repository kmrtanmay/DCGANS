import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model import Generator,Discriminator
from data_loader import get_testdata, get_traindata
from transformer import transformer


manualseed = 100
torch.manual_seed(manualseed)

#device configuaration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


#Hyperparameters
image_size = 64
batch_size = 128
num_z = 100                # length of the noise vector input to the generator i.e the size of the latent vector
num_channels = 3           # num of channels in the training image
num_gf = 64                # size of feature maps in generator
num_df = 64                # size of feature maps in discriminator
lr = 0.0002                # Learning rate for optimisers
beta1 = 0.5                # Beta1 hyperparameter for Adam optimizers
num_epochs = 5

#Image Transformations
transform = transformer(image_size)
#Loading the train data 
trainloader = get_traindata(transform,batch_size)

# Plot some training images
real_batch = next(iter(trainloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# This function takes an initialized model as input and reinitializes all
# convolutional, convolutional-transpose, and batch normalization layers.
# all model weights are randomly initialized from a Normal 
# distribution with mean=0, stdev=0.02

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)



# Create the generator
generator = Generator()
generator.to(device)

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
generator.apply(weights_init)

#Print Model
print(generator)


# Create the Discriminator
discriminator = Discriminator()
discriminator.to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
discriminator.apply(weights_init)

# Print the Model
print(discriminator) 


# Initialize BCELoss function
criterion = nn.BCELoss()      # Binary Cross Entropy Loss

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, num_z, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))


# Lists to keep track of progress
img_list = []
Gen_losses = []
Dis_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the trainloader
    for i, data in enumerate(trainloader, 0):

        ############################
        # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_batch = data[0].to(device)
        b_size = real_batch.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through Discriminator
        output = discriminator(real_batch).view(-1)
        # Calculate loss on all-real batch
        DisLoss_real = criterion(output, label)
        # Calculate gradients for Discriminator in backward pass
        DisLoss_real.backward()
        Dis_x = output.mean().item()


        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, num_z, 1, 1, device=device)
        # Generate fake image batch with Generator
        fake_batch = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with Discriminator
        output = discriminator(fake_batch.detach()).view(-1)
        # Calculate Discriminator's loss on the all-fake batch
        DisLoss_fake = criterion(output, label)
        # Calculate the gradients for this batch
        DisLoss_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        DisLoss = DisLoss_real + DisLoss_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through Discriminator
        output = discriminator(fake_batch).view(-1)
        # Calculate G's loss based on this output
        GenLoss = criterion(output, label)
        # Calculate gradients for Generator
        GenLoss.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     DisLoss.item(), GenLoss.item(), Dis_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        Gen_losses.append(GenLoss.item())
        Dis_losses.append(DisLoss.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
            with torch.no_grad():
                fake_batch = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_batch, padding=2, normalize=True))

        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(Gen_losses,label="G")
plt.plot(Dis_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

real_batch = next(iter(trainloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


# Save and load the generator model.
torch.save(generator, 'generator.ckpt')
generator = torch.load('generator.ckpt')

# Save and load only the generator parameters (recommended).
#torch.save(generator.state_dict(), 'params_gen.ckpt')
#generator.load_state_dict(torch.load('params_gen.ckpt'))

# Save and load the discriminator model.
torch.save(discriminator, 'discriminator.ckpt')
discriminator = torch.load('discriminator.ckpt')

# Save and load only the generator parameters (recommended).
#torch.save(discriminator.state_dict(), 'params_dis.ckpt')
#discriminator.load_state_dict(torch.load('params_dis.ckpt'))