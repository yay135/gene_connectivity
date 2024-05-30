export FILEID='1fTHAV0yL19S6MxFfa_UN8-ug75T944OP'
export FILENAME='project_gene_connectivity_data.zip'

wget https://bootstrap.pypa.io/get-pip.py

python -m pip install gdown
gdown $FILEID

echo Download complete. Extracting ... 