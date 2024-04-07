#pip install pytorch-lightning

#pip install wandb
                                                               # Import necessary libraries
import pytorch_lightning as L
from torchvision import transforms, models,datasets
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader ,random_split,Subset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy
import torch.nn.functional as F
import torch
import argparse
#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
#import wandb
from Lightning_CNN_Trainer import Lightning_CNN
from Data_manager import root_dataset,inaturalist_train,inaturalist_val,inaturalist_test
def save_images_with_labels(images, true_labels, predicted_labels,label_class, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(true_labels)):
        true_labels[i]=label_class[true_labels[i]]
        predicted_labels[i]=label_class[predicted_labels[i]]
    for i, (image, true_label, predicted_label) in enumerate(zip(images, true_labels, predicted_labels)):
        # Plot the image with labels
        plt.imshow(image.permute(1, 2, 0).numpy())  # Assuming image tensor is in CHW format
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')# Save the image with labels
        filename = os.path.join(output_dir, f"image_{i}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.show()
def main(args):     # Main function
    if 5==5:
        #config=wandb.config    #Extracting configuration parameters from arguments
        #wandb.run.name = 'bs-'+str(config.batch_size)+'-lr-'+ str(config.learning_rate)+'-ep-'+str(config.epochs)+ '-op-'+str(config.optimizer)+ '-dls-'+str(config.dense_layer_size)+ '-act-'+str(config.activation)+'-do-'+str(config.dropout)+'-bn-'+str(config.batch_normalization)+'-cs-'+','.join(str(x) for x in config.conv_attributes_channels)+'-ck'+','.join(str(x) for x in config.conv_attributes_kernel_size)+'-pk-'+','.join(str(x) for x in config.pool_attributes_kernel_size)+'-ps-'+','.join(str(x) for x in config.pool_attributes_stride)
        number_of_filter=[[256,128,64,32,16],[32,64,32,64,32],[32,32,32,32,32],[16,32,64,128,256],[64,64,64,64,64]]
        layers=number_of_filter[args.number_of_filter_per_layer]
        kernel_size_per_layer=[[3,3,5,7,9],[7,5,5,3,3],[11,7,5,3,3],[3,3,3,5,5],[3,3,3,3,3],[11,7,7,5,3],[3,5,7,9,11]]
        kernel_size=kernel_size_per_layer[args.kernel_size_per_layer]
        pooling_kernel_size_per_layer=[[2,2,2,2,2],[2,2,2,1,1],[2,1,3,1,2],[3,3,3,2,2]]
        pool_kernel=pooling_kernel_size_per_layer[args.pooling_kernel_size_per_layer]
        pooling_stride_size_per_layer=[[2,2,2,2,2],[2,2,2,1,1],[1,1,2,2,2],[1,2,1,2,1],[2,2,2,2,1]]
        pool_stride=pooling_stride_size_per_layer[args.pooling_stride_size_per_layer]
        batch_normalization=args.batch_normalization
        drop_out=args.dropout_rate
        activation_function=args.activation
        optimizer=args.optimizer
        b_size=args.batch_size
        dense_layer_output=args.hidden_size
        epoch=args.epochs
        learning_rate=args.learning_rate
    #aug_bit=True              
        i_d=224        # Calculate input size for fully connected layer
        D=0
        for i in range(5):
            D = (i_d - kernel_size[i])+3
            D = (D - pool_kernel[i])//pool_stride[i] + 1
            i_d = D
        root_obj=root_dataset(args.path)      # Create datasets and dataloaders
        train_data=root_obj.get_train_data()
        val_data=root_obj.get_val_data()
        dataset1=inaturalist_train(train_data)
        dataset2=inaturalist_val(val_data)
        dataset3=inaturalist_test(args.path)
        test_dataloader=DataLoader(dataset=dataset3,batch_size=8,shuffle=False,num_workers=1)
    #print(len(dataset1))
    #print(len(dataset2))
        #wandb_logger = WandbLogger(project='amar_cs23m011', entity='Assignment2-CS6910')
        dataloader=DataLoader(dataset=dataset1,batch_size=b_size,shuffle=True,num_workers=2)
        val_dataloader=DataLoader(dataset=dataset2,batch_size=b_size,shuffle=False,num_workers=2)
        # Create model
        model=Lightning_CNN(layers,kernel_size,pool_kernel,pool_stride,(D**2)*layers[4],batch_normalization,drop_out,activation_function,optimizer,dense_layer_output,learning_rate)
        trainer = L.Trainer(accelerator='auto',devices="auto",max_epochs=epoch)   # Create trainer and fit model
        trainer.fit(model,dataloader,val_dataloader)
        print_pic=0
        if print_pic:
            model.eval()
            true_label=[]
            pred_label=[]
            images=[]
            c=30
            for im,label in test_dataloader:
                if c==0:
                    break;
                with torch.no_grad():
                  output = model(im)
                _, predicted_labels = torch.max(output, 1)
                true_label.append(label)
                pred_label.append(predicted_labels)
                images.append(im.squeeze(0))
                c=c-1
# Process the output as needed (e.g., getting predicted labels)
            image1=torch.stack(images, dim=0)
            c=0
            for i in range(30):
                if(pred_label[i]==true_label[i]):
                    c=c+1
            print("Plotting_image _accuracy "+c/30)
            labels_class=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Replilia']
            save_images_with_labels(image1, true_label, pred_label,label_class,args.path)
                
        
        test_dataloader=DataLoader(dataset=dataset3,batch_size=8,shuffle=False,num_workers=1)     # Test the model
        trainer.test(dataloaders=test_dataloader)

if  __name__ =="__main__":
  
    parser = argparse.ArgumentParser()  #taking arguments from command line arguments
    parser.add_argument('-p','--path',type=str,help='provide the path where your data is stored in memory,Read the readme for more description')
    parser.add_argument('-e','--epochs',type=int,default=15,help='Number of epochs to CNN')
    parser.add_argument('-b','--batch_size',type=int,default=16,help='Batch size used to train CNN')
    parser.add_argument('-o','--optimizer',type=str,default='adam',choices=['sgd','adam','nadam'],help='optimzer algorithm to evaluate the model')
    parser.add_argument('-a','--activation',type=str,default='gelu',choices=['relu','gelu','selu','elu'],help='activation function used in the model')
    parser.add_argument('-sz','--hidden_size',type=int,default=128,help='Number of neuron in fully connected layer used in convolutional neural network')
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
    parser.add_argument('-do','--dropout_rate',type=float,default=0.3,choices=[0.2,0.3,0.4],help='drop out rate to regularilze the model')
    parser.add_argument('-bn','--batch_normalization',type=bool,default=True,choices=[True,False],help='wheather you want to branch normalize the layers')
    parser.add_argument('-nf','--number_of_filter_per_layer',type=int,default=0,choices=[0,1,2,3,4],help='from ReadMe file read the Filter configuration index and press')
    parser.add_argument('-ks','--kernel_size_per_layer',type=int,default=0,choices=[0,1,2,3,4,5,6],help='from ReadMe file read the kernel_size configuration index and press')
    parser.add_argument('-pk','--pooling_kernel_size_per_layer',type=int,default=1,choices=[0,1,2,3],help='from ReadMe file read the pooling_kernel_size configuration index and press')
    parser.add_argument('-ps','--pooling_stride_size_per_layer',type=int,default=1,choices=[0,1,2,3],help='from ReadMe file read the pooling_stride_size configuration index and press')
    args = parser.parse_args()
    main(args)
