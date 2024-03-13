# Project Gene Connectivity
## 1. Introduction

![alt text](https://github.com/yay135/gene_connectivity/blob/master/arch_gene_connectivity.png?raw=true)

This project aims to map gene connectivities using different deep-learning models such as MLP, Autoencoders, and GNNs with additional gene edges. Our primary contribution is the Introduction of GNNs into gene mapping as shown above. Due to the unique abilities of GNNs to utilize nodes and edges, we can incorporate more types of biological information resulting in an overall much better mapping performance.  

Gene connectivities are mapped extensively through GNNs with added edges such as biological pathways, correlations, regulations, PPIs, etc. After the models were built, we compared the model performances and found that different gene edges result in drastically different mapping qualities in GNNs, some edges are more useful than others. Between models, some models show much stronger performance against other models suggesting high task specificities in gene mapping.

## Docker builds of our best models are now available!
### Install Docker
Install Docker using the following link:  
https://docs.docker.com/engine/install/  
Recommended System specs ubuntu 20.04 LTS or better with 24GB RAM or more with at least 30GB disk space.  
### Run the models  
These are CPU-only builds.  
Pull our docker images:  
``docker pull yay135/gcn_con``  
OR  
``docker pull yay135/pna_con``  

Arrange your expression data files in a single folder such as /path/to/exp/  
Run the GCN model, replace $exp_path with your path to your expression files folder:  
``docker run --rm -v $exp_path:/workspace yay135/gcn_con``    
OR Run the GCN model:  
``docker run --rm -v $exp_path:/workspace yay135/pna_con``  

By default, the model outputs scaled gene expressions, if you need to output in tpm format:  
``docker run --rm -v $exp_path:/workspace -e OUT_TPM='True' yay135/gcn_con``  
OR  
``docker run --rm -v $exp_path:/workspace -e OUT_TPM='True' yay135/pna_con``    

The commands will scan the your /path/to/exp/ and infer from all of files with ".csv" extension. For each csv file in your folder, you must have **ENSG######### (ensemble ids with no versions) as headers**, the headers must contain enough (at least 30%) of the predictor genes, the expression value must be **RNA-seq TPM**. Please refer to predictors.csv and inferred.csv for predictor and inferred genes. The genes are extracted for gtex_tcga_normal dataset.

The GCN and PNA models are build with the following edges: GCN: 3D inter chromosome, Pathway, 1D genomic, PNA: Correlation, 3D inter chromosome, 3D intra chromosome, Pathway, 1D genomic. **The Models are built for normal tissues**.

**The following tutorials are for non-docker usages.**

## 2. Data and model
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
A CUDA 11.6 compatible NVIDIA GPU with 48 GB or more VRAM. (If you are not able to meet the hardware requirements consider configure a cpu-only environment using the 3.4.2 instructions bellow)  


### 3.2 Install the Required Software
The following softwares are required if you wish to use GPU:  
Ubuntu 20.04+  
bash @ lastest   
wget @ lastest  
Anaconda @ 23.5.2  
NVIDIA GPU driver for Linux 510.47.03  
CUDA 11.6  

The following softwares are required if you wish to use cpu only:  
Ubuntu 20.04+  
bash @ lastest   
wget @ lastest  
Anaconda @ 23.5.2  

The rest of the softwares such as pytorch are configured automatically using conda or pip, see section 3.4.

GPU or CPU selection is automated at the runtime.

### 3.3 Clone the current project
Run the following command to clone the project.  
``git clone https://github.com/yay135/gene_connectivity.git``  
### 3.4 Configure environment
#### 3.4.1 Configure for gpu
Change the directory to the project root folder.  
``cd gene_connectivity``  

Run the following command to create a conda environment automatically.  
``conda env create -f gnn_cuda.yml``  

Activate the environment.  
``conda activate gnn_cuda``   
#### 3.4.2 Configure for cpu
If you don't have a GPU or if your system specs are different than specified,  
configure a CPU-only environment using the following commands:  
``cd gene_connectivity``  
``conda create -n gnn python=3.10``  
``conda activate gnn``  
``python -m pip install -r requirements.txt``  

### 3.5 Data and output
#### 3.5.1 Data 
Run the following command to initialize the required data:  
``python init_data.py``

#### 3.5.2 Data types
The data includes expression data and edge data, expression data is first log2 transformed and then MinMax scaled. Edge data is mapped to the index of genes(columns). For simplicity and clarity, the code to process the data and the edges are not provided. They are in folder "gtex_tcga_normal", "tcga_ccle_bc" and "tcga_cptac_bc". The first word indicates training data source, the second word indicates testing data source and the third word indicates whether the data is normal or breast cancer. For example, "gtex_tcga_normal" indicates training on gtex normal dataset and testing on tcga normal dataset.  

The X.csv is the training predictor gene expression dataset.  
The y.csv is the training inferred gene expression dataset.  
The X_val.csv is the testing predictor gene expression dataset.  
The y_val.csv is the testing inferred gene expression dataset.  

The edge datasets are in folders ending with "edges". The first column is the source of the edge, the second column is the target of the edge. For example, in pathway_gtex_tcga_normal.csv 3249,15423 indicates the 3249th gene(column) (X.csv , y.csv horizontally combined) is pointing the 15423rd gene(column) in dataset gtex_tcga_normal, each column is a gene and is 0 indexed.  

#### 3.5.3 Output
The testing is automatically initialized after the training. The output is inferred gene expressions. They will appear in the folder model_out with model type, edge type, and training, and testing data information attached to the file name.

### 3.6 Example tasks
CD into the root folder, and modify the configure.py to run different tasks.  
Options:  
--help show all available options.    
-m/--model specify model type. Required.  
-d/--data specify dataset configuration. Required.  
-e/--edges specigy edges you wish to input to pna or gcn. Required if model is pna or gcn.   
-k/--mask specify whether to use masked datasets. Optional, default is false.  
-v/--validation specify whether to run validation test after training. Optional, default is true.   



#### 3.6.1 Example 1
``python run.py -m auto_encoder -d tcga_cptac_bc``  
This will train and test an autoencoder on the tcga and cptac breast cancer dataset. Edges are ignored if model_type is not set to 'gcn' or 'pna'.
#### 3.6.2 Example 2
``python run.py -m pna -d tcga_ccle_bc -e cor hic_inter pathway string dorothea``  
This will train and test a pna model on tcga ccle breast cancer dataset and with additional edges such as correlations, PPI, Hic edges, and Pathway edges.

#### 3.6.2 Example 3
``python run.py -m gcn -d gtex_tcga_normal -e hic_intra spatial -k true -v false``  
This will train a gcn model on the gtex dataset using edges from intra-chromo hic data and 1d spatial data, also, testing/validation is ignored.


## Team
If you have any questions or concerns about the project, please contact the following team member:
Fengyao Yan fxy134@miami.edu 
