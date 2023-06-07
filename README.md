# How to run Spatial ID 

before you start to run the program, you should make something installed.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
pip install torch-geometric==2.0
```
Before running, you need to copy the `cell_type_annotation_model.pyc` file to the program running directory. The first is the introduction of the package

```python
import os
import time
from scipy.sparse import *
import random
import argparse
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch_geometric
from cell_type_annotation_model import DNNModel, SpatialModelTrainer
import torch.utils.data as Data
```

Then there is the setting of random seed, so that the experimental results can be repeated

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Next is the reading of single cell data. It should be noted that the homogenization of single cells needs to be consistent with the spatial group. If the. X matrix of a single cell is the original counts matrix, then the. X matrix of the spatial group should also be the original matrix; If the. X matrix of a single cell is a matrix processed by log normalization, then the. X matrix of a spatial group should also be a matrix processed by log normalization, the same applies to other matrices. I have already processed the single cell data and spatial group data consistently in this note, so I will not elaborate on the normalization steps here.

```python
adata = sc.read('../single_cell/snrna_fetal_brain.h5ad')
```

Next is the training of the DNN model. We can see the process in the first diagram of the Spatial ID original text. The first step is to train a DNN model using one's own single cell data, and then proceed with subsequent processing. The following is the training process of the DNN model I wrote for your reference

```python
def dnn_train(adata, labels, hidden_dim, train_percent, epoches, batch_size, learning_rate, model_save_path):
    
    #### labels_transform
    def labels_transform(raw_labels):
        raw_labels_copy = raw_labels.copy()
        labels_categroies = [i for i in set(raw_labels)]
        length = len(labels_categroies)
        re_dic = {}
        for i in range(length):
            raw_labels_copy.replace(labels_categroies[i],i, inplace = True)
            re_dic[i] = labels_categroies[i]
        return [i for i in raw_labels_copy],re_dic
    
    ### save models
    def save_model(n):
        print('\n==> save model...{}'.format(n))
        torch.save({
            'model':DnnModel,
            'label_names': [lable_dic[i] for i in lable_dic],
            'marker_genes':gene_names,
        }, model_save_path+ '_'+ str(n) +'.dnnmodel')
    
    
    ###preprocess
    print('\n==> Preprocessing...')
    adata_x = adata.X.todense()
    raw_labels = adata.obs[labels]
    gene_names = [i for i in adata.var_names]
    labels, lable_dic = labels_transform(raw_labels)
    inputs, outputs = torch.from_numpy(np.array(adata_x)).to(torch.float32), torch.from_numpy(np.array(labels))

    ###dataloader
    print('\n==> Predataloader...')
    dataset = Data.TensorDataset(inputs, outputs)
    tran_legth, test_length = int(len(dataset)*train_percent), len(dataset) - int(len(dataset)*train_percent)
    train_set, test_set = Data.random_split(dataset, [tran_legth, test_length])
    train_loader = Data.DataLoader(train_set, 
                                   batch_size=batch_size, 
                                   shuffle=True,)
    
    test_loader = Data.DataLoader(test_set, 
                                  batch_size=batch_size, 
                                  shuffle=False)
    
    #####
    print('\n==> DnnModel...')
    input_dim = inputs.shape[1]
    hidden_dim = hidden_dim
    output_dim = len(set(labels))
    learning_rate = learning_rate
    
    DnnModel = DNNModel(input_dim = input_dim, 
                    hidden_dim = hidden_dim, 
                    output_dim = output_dim,
                    drop_rate=0.2
                       )
    ###train
    print('\n==> train...')
    counts = 0
    max_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(DnnModel.parameters(), lr = learning_rate)
    for epoch in range(epoches):
        for input_tensor,transformed_labels in train_loader:
            out = DnnModel(input_tensor)
            loss = criterion(out, transformed_labels)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        
        correct = 0
        total = 0
        with torch.no_grad(): 
            for input_tensor,transformed_labels in test_loader:
                pred = DnnModel(input_tensor).argmax(1)
                
                correct += (pred == transformed_labels).sum().item()
                total += transformed_labels.size(0)
        acc = correct / total
        print('[epoch {}], loss: {}, acc：{}'.format(epoch + 1, loss.data, correct / total))
        
        if(acc > max_acc):
            max_acc = acc
            save_model(counts)
            counts += 1
    
    save_model(counts)
```

Label is the single cell annotation in your OBS, train_ Percentage is the percentage of data you use for training, epochs is the number of training sessions, learning_ Rate is the learning rate of training, which depends on one's own tuning; Model_ Save_ Path is the storage location for the DNN model that I have trained myself. I have set an accuracy parameter here, and an increase in acc will automatically save this model. Finally, a model will also be saved. It is recommended to train the acc for this step to over 80%. If the acc is too low, it indicates that there may be a problem with single cell annotation.

```python
dnn_train(adata, 
          labels = 'label', 
          hidden_dim = 800, 
          train_percent = 0.8, 
          epoches = 200, 
          batch_size = 256, 
          learning_rate = 0.01, 
          model_save_path = './dnnmodel/model')
```

Next comes the other steps in the Spatial ID paper, besides the DNN model training, which is the transfer process. The code is as follows

```python
def transfer(adata, model, transfer_batchsize, result_csv, new_adata, device,k_graph,params):
    
    # Load DNN model trained by sc-dataset.
    checkpoint = torch.load(model)
    dnn_model = checkpoint['model'].to(device)
    # Initialize DNN input.
    marker_genes = checkpoint['marker_genes']
    gene_indices = adata.var_names.get_indexer(marker_genes)
    adata_X = np.pad(adata.X.todense(), ((0,0),(0,1)))[:, gene_indices]
    norm_factor = np.linalg.norm(adata_X, axis=1, keepdims=True)
    norm_factor[norm_factor == 0] = 1
    dnn_inputs = torch.Tensor(adata_X / norm_factor).split(transfer_batchsize)
    
    # Inference with DNN model.
    dnn_predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dnn_inputs):
            inputs = inputs.to(device)
            outputs = dnn_model(inputs)
            dnn_predictions.append(outputs.detach().cpu().numpy())
    label_names = checkpoint['label_names']
    adata.obsm['pseudo_label'] = np.concatenate(dnn_predictions)
    adata.obs['pseudo_class'] = pd.Categorical([label_names[i] for i in adata.obsm['pseudo_label'].argmax(1)])
    adata.uns['pseudo_classes'] = label_names
    print(set(adata.obs['pseudo_class']))
    # Construct spatial graph.
    gene_mat = torch.Tensor(adata_X)
    cell_coo = torch.Tensor(adata.obsm['spatial'])
    data = torch_geometric.data.Data(x = gene_mat, pos = cell_coo)
    data = torch_geometric.transforms.KNNGraph(k = k_graph, loop=True)(data)
    data.y = torch.Tensor(adata.obsm['pseudo_label'])
    # Make distances as edge weights.
    data = torch_geometric.transforms.Distance()(data)
    data.edge_weight = 1 - data.edge_attr[:,0]
    
    # Train self-supervision model.
    params = params
    input_dim = data.num_features
    num_classes = len(adata.uns['pseudo_classes'])
    trainer = SpatialModelTrainer(input_dim, 
                                  num_classes, 
                                  device, 
                                  params = params)
    trainer.train(data, params)
    # trainer.save_checkpoint()
    
    # Inference.
    print('\n==> Inferencing...')
    predictions = trainer.valid(data)
    celltype_pred = pd.Categorical([adata.uns['pseudo_classes'][i] for i in predictions])
    result = pd.DataFrame({'cell': adata.obs_names.tolist(), 'celltype_pred': celltype_pred})
    result.to_csv(result_csv, index=False)
    adata.obsm['celltype_prob'] = predictions
    adata.obs['celltype_pred'] = pd.Categorical(celltype_pred)
    adata.write(new_adata)    
    sc.pl.spatial(adata, img_key=None, color=['celltype_pred'], palette  =  'gnuplot2', spot_size=30, show=True)
```

The above is the definition of the function, and the following are the parameters and corresponding operations. Fill in the DNN model you trained above with the model, result_ CSV stores the cell type data corresponding to each bin/cell, new_ Adata is the data annotated with cells, and params is the parameter that needs to be adjusted. You need to debug it yourself based on the situation of the transfer.

```python
#Read in spatial group data

spatial_adata = sc.read('../spatial/A5.h5ad')
params = {
    'pca_dim': 100,  
    'k_graph': 30,
    'edge_weight': True,
    'kd_T': 3,
    'feat_dim': 300,
    'w_dae': 10.0,
    'w_gae': 5.0,
    'w_cls': 50.0,
    'epochs': 30,
    'lr': 0.0005,
}

transfer(spatial_adata, 
         model = 'test___11.dnnmodel', 
         transfer_batchsize = 2048, 
         result_csv = 'test_10.csv', 
         new_adata ='new_10.h5ad', 
         device = 'cpu',
         k_graph = 30,
         params = params)
```
