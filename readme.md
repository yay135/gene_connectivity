# Project Gene Connectivity
## 1. Introduction

![alt text](https://github.com/yay135/gene_connectivity/blob/master/arch_gene_connectivity.png?raw=true)

This project aims to map gene connectivities using different deep-learning models such as MLP, Autoencoders, and GNNs with additional gene edges. Our primary contribution is the Introduction of GNNs into gene mapping as shown above. Due to the unique abilities of GNNs to utilize nodes and edges, we can incorporate more types of biological information resulting in an overall much better mapping performance.  

Gene connectivities are mapped extensively through GNNs with added edges such as biological pathways, correlations, regulations, PPIs, etc. After the models were built, we compared the model performances and found that different gene edges result in drastically different mapping qualities in GNNs, some edges are more useful than others. Between models, some models show much stronger performance against other models suggesting high task specificities in gene mapping.

## 2. Data and model availability
Datasets are fully released. Due to the large size, model files are not released in this repository. This repository supports complete training and testing with the following models and options:  
### 2.1 Available models
MLP  
Autoencoder  
LinearRegression
GCN
PNA
### 2.2 Available datasets
Training on GTEx normal, testing on TCGA normal    
Training on TCGA BC, testing on CPTAC BC  
Training on TCGA BC, testing on CCLE BC  
### 2.3 Available edges 
Correlation  
String (PPI)  
Dorothea (TF)  
Hi-C inter-chromosome (3D)  
Hi-C intra-chromosome (3D)  
Pathway  
Spatial (1D)   
## 3. How to use this repo
### 3.1 System requirements
Due to the immense connections and edges in a gene map (graph), the requirements to train a GCN or PNA to map 20000 genes are high.
You must have at least 100GB free disk space, at least 64GB usable RAM size, and a 4-core CPU with AVX support.  
A CUDA 11.6 compatible NVIDIA GPU with 48 GB or more VRAM.  
Ubuntu 16.04+  

### 3.2 Install the Required Software
Anaconda 23.5.2  
NVIDIA GPU driver for Linux 510.47.03  
CUDA 11.6  

Additionally, make sure your system has bash and wget installed.
### 3.3 Clone the current project
Run the following command to clone the project.  
``git clone https://github.com/yay135/gene_connectivity.git``  
### 3.4 Configure environment
#### 3.4.1 
Change the directory to the project root folder.  
``cd gene_connectivity``  

Run the following command to create a conda environment automatically.  
``conda env create -f gnn_cuda.yml``  

Activate the environment.  
``conda activate gnn_cuda``   
#### 3.4.2
If you don't have a GPU or if your system specs are different than specified, configure a CPU-only environment as follows:  
Install Python 3.11 and pip on your system.  
Run the following command to install all the required Python libraries.  
``python -m pip install -r requirements.txt``  

### 3.5 Run training and validation
#### 3.5.1 Data 
Run the following command to initialize the required data:  
``python init_data.py``

#### 3.5.2 Data types
The data includes expression data normalized and MinMax scaled. They are in folder "gtex_tcga_normal", "tcga_ccle_bc" and "tcga_cptac_bc". The first word indicates training data source, the second word indicates testing data source and the third word indicates whether the data is normal or breast cancer. For example, "gtex_tcga_normal" indicates training on gtex normal dataset and testing on tcga normal dataset.  

The X.csv is the training predictor gene expression dataset.  
The y.csv is the training inferred gene expression dataset.  
The X_val.csv is the testing predictor gene expression dataset.  
The y_val.csv is the testing inferred gene expression dataset.  

The edge datasets are in folders ending with "edges". The first column is the source of the edge, the second column is the target of the edge. For example, in pathway_gtex_tcga_normal.csv 3249,15423 indicates the 3249th column (X.csv , y.csv horizontally combined) is pointing the 15423rd column as an edge in dataset gtex_tcga_normal, each column is a gene and is 0 indexed.  

#### 3.5.3 Output
The testing is automatically initialized after the training. The output is inferred gene expressions. They will appear in the folder model_out with model type, edge type, and training, and testing data information attached to the file name.

### 3.6 Example tasks
CD into the root folder, and modify the configure.py to run different tasks.
### 3.6.1
Modify configure.py the enclosed lines as follows

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
model_type = 'auto_encoder'  
mask_exp = False  
fd = 'tcga_cptac_bc'  
edges = ["cor", "string", "dorothea", "hic_intra", "pathway", "spatial", "hic_inter"]  
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  

then 

``python run.py``

This will train and test an autoencoder on the tcga and cptac breast cancer dataset. Edges are ignored if model_type is not set to 'gcn' or 'pna'.
### 3.6.2
Modify configure.py the enclosed lines as follows

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
model_type = 'pna'  
mask_exp = False  
fd = 'tcga_ccle_bc'  
edges = ["cor", "string", "hic_intra", "pathway"]  
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  

then 

``python run.py``

This will train a pna model on tcga ccle breast cancer dataset and with additional edges such as correlations, PPI, Hic edges, and Pathway edges.

## Team
If you have any questions or concerns about the project, please contact the following team member:
Fengyao Yan fxy134@miami.edu 
