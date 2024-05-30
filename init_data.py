import subprocess
subprocess.run(['bash', 'gdownload.sh'])
# importing the zipfile module 
from zipfile import ZipFile 
  
# loading the temp.zip and creating a zip object 
with ZipFile("project_gene_connectivity_data.zip", 'r') as zObject: 
  
    # Extracting all the members of the zip  
    # into a specific location. 
    zObject.extractall( 
        path=".") 


print("data init successful!")