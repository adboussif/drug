# Computer-Assisted Drug Design (CADD) Learning Journey

This repository documents my self-directed learning journey in Computer-Assisted Drug Design (CADD). As I delve into the fascinating intersection of machine learning, deep learning, and bioinformatics for drug discovery, I will be publishing various notebooks covering different CADD methodologies and topics.

## About This Project

Drug discovery is a complex, time-consuming, and expensive process. Computer-Assisted Drug Design (CADD) offers powerful computational approaches to accelerate and refine the development of new therapeutic molecules. This project aims to build a comprehensive understanding and practical application of CADD pipelines.

My learning focuses on several key areas within CADD:

* **Understanding Drug Discovery Fundamentals**: Grasping the economic and scientific challenges of drug discovery, including the identification of viable candidate molecules and early-stage filtering.
* **Molecular Representation and Manipulation**: Learning to encode molecules using formats like SMILES, generate molecular fingerprints (e.g., ECFP), and measure molecular similarity (e.g., Tanimoto coefficient) using libraries such as RDKit.
* **Machine Learning for Property Prediction**: Training supervised models (e.g., linear regressions, random forests) to predict crucial target properties like solubility, toxicity, and enzyme inhibition (e.g., cytochrome P450) based on public datasets like ChEMBL and PubChem.
* **Exploratory Data Analysis**: Applying techniques like Principal Component Analysis (PCA) to reduce dimensionality and explore chemical space diversity.
* **Deep Learning for De Novo Design**: Exploring generative models (variational autoencoders, graph neural networks) using libraries like PyTorch and DeepChem to design novel molecular structures with optimized bioactivity or pharmacokinetic profiles.
* **Virtual Screening (VS)**:
    * **Ligand-Based Virtual Screening (LBVS)**: Employing molecular similarity searches and Machine Learning models (QSAR/QSPR) to identify active molecules when the target protein's 3D structure is unknown. This includes preparing data, training classification models (like Random Forest), and evaluating their performance using metrics such as PR AUC and MCC.
    * **Structure-Based Virtual Screening (SBVS)**: Utilizing molecular docking to predict how potential ligands bind to a known 3D protein target, assessing binding poses and affinity.
* **ADMET Property Prediction**: Developing computational models (QSPR and other AI models) to predict Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties early in the discovery phase, reducing late-stage failures. This includes predicting aqueous solubility, CYP inhibition, and hERG channel blockade.

## Project Structure

This repository will contain Jupyter notebooks, Python scripts, and potentially data files organized by topic or methodology.

* `notebooks/`: Contains Jupyter notebooks demonstrating various CADD techniques and experiments.
* `data/`: (To be added if necessary) Will store datasets used in the notebooks.
* `scripts/`: (To be added if necessary) Utility scripts.

## Getting Started

To run the notebooks in this repository, you will need to have Python installed along with the necessary libraries.

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/CADD-Learning-Journey.git](https://github.com/yourusername/CADD-Learning-Journey.git)
    cd CADD-Learning-Journey
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file will be generated as I add more notebooks with specific library dependencies. For now, you will likely need `rdkit-pypi`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch` (if using deep learning), and `deepchem`.)

### Running the Notebooks

1.  Start Jupyter Notebook/Lab:
    ```bash
    jupyter notebook
    ```
2.  Navigate to the `notebooks/` directory and open any `.ipynb` file to explore the content.


**Note**: This README will be continuously updated as my learning progresses and new notebooks are added.