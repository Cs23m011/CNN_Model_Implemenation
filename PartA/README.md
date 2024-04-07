# CS6910_assignment1
please install the dependencies before running the program other wise it may give error .I have added the requirement.txt file in the same folder. Please download the code from github and extract it .Then install the dependencies and run train.py .Note that all the other necessary module should be there in the same folder as with train.py as it import all the other classes from different file .
```
pip install requirement.txt

python train_cnn.py -p path. Providing path is mandatory . provide the full path of the directory where your train and test data stored. only give input till /train or
/val. I mean give input till  C:\Users\USER\Downloads\nature_12K\inaturalist_12K not this full C:\Users\USER\Downloads\nature_12K\inaturalist_12K\train .
```
My code is very much flexible to add in command line arguments . I am adding the list of possible argument below for your reference.Please try to run this on local PC or from command promt by ensuring all the libraries in requirements.txt are already installed in your system. Because in google colab this might give problem .

| Name        | Default Value   | Description |
| ------------- |:-------------:| -----:|
| `-e,--epochs` | 15      |    number of epochs your algorithm iterate |
|`-b,--batch_size`|16      |batch size your model used to train |
|`-o,--optimizer`|adam|Choices=['sgd','adam','nadam']|
|`-a, --activation`|relu|choices=['relu','gelu','selu','elu']|
|`-sz,--hidden_size`|128|Number of neuron in each layer|
|`-lr,--learning_rate`|0.001|Learning rate used to optimize model parameters|
|`-we,--wandb_entity`|amar_cs23m011|Project name used to track experiments in Weights & Biases dashboard|
|`-do,--drop_out`|0.3|drop out rate for the project|
|`-bn,--batch_normalization`|True|batch normalization to be done or not|
|`-nf,--number_of_filter_per_layer`|0|[0,1,2,3,4]|choose any of the configuration filter from configuration array|

Few example are shown below to how to give inputs:-
```
python train.py
```
This will run my best model which i get by validation accuracy. after that it will create a log in a project named CS6910-assignment12 by default until user dont specify project name.
```
config = {
        "epochs" : 20,
        "batch_size": 128,
        "loss": "cross_entropy",
        "optimizer": "adam" ,
        "learning_rate": 0.0001 ,
        "num_layers": 4 ,
        "hidden_size": 256,
        "activation" : "tanh",
        "weight_init_method" : "He_normal" ,
        "weight_decay": 0

 }
```
Now if you want to change the number of layer I just have to execute the following the command.
```
python train.py -nhl 5
```
this will change the number of layer to 5. Similarly we can use other commands as well.
