# ApisTox-Data-Analysis
A surface level analysis of the ApisTox dataset in an effort to determine a potential link between agrochemicals and their different properties, categories, and toxicity to bees. 

# Libraries and Software Used
- Python, Pandas, NumPy, Matplotlib, Sklearn, Seaborn
- Jupyter Notebook

# Key Insights
- Discovered that agrochemicals have been getting less toxic accross the board in the last few decades after reaching an all-time high around 1990. Additionally, earlier pesticides were not as toxic as current pesticides according to the visualization and limited data.
- Measurements comparable to LD50 for humans should be found for insects that are adversely affected for a more accurate analysis of agrochemical compounds. PPDB level was sufficient for a surface level analysis but more precise results require more precise measurements that can be derived from a regression analysis rather than a classification analysis. Unfortunately there is a lack of data regarding this in the dataset.
- Recommend using molecular structure in the form of SMILES to help perform classification tasks to more accurately predict the toxicity of agrochemical compounds that aren't part of the dataset.

# How to run
- Download repository
- Navigate to directory where files are located
- Install dependencies with pip install -r requirements.txt
- Run python file with python [filename].py
