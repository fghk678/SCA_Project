This is the code repository for reproducing the work in the paper "Identifiable Shared Component Analysis of Unpaired Multimodal Mixtures" [1].

[1] Subash Timilsina, Sagar Shrestha, and Xiao Fu. Identifiable shared component analysis of unpaired multimodal mixtures. Advances in neural information processing systems, 37, 2025.
# Instructions

## Prerequisites

1. **Install Python**

   Ensure you have Python version 3.8.18 installed on your system.

2. **Install Required Packages**

   Use the following command to install all required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Faiss-GPU (for Word embedding alignment)**
   - GPU users are recommended to install Faiss-GPU for faster nearest neighbors computation:
     ```bash
     conda install faiss-gpu -c pytorch
     ```

---

## Steps to Reproduce Experiments

### 1. Synthetic Data Experiments

- **Theorem 1 Validation**:
  Run `python syn_theorem1.py` to validate **Theorem 1**.
  
- **Theorem 2 Validation**:
  Run `python syn_anchor.py` to validate **Theorem 2**.

#### Newly developed code

The newly developed code includes `syn_theorem1.py` and `syn_anchor.py` based on the orignal jupyter notebook code `synthetic_train.ipynb`, along with other debugging scripts such as `syn_cpu_test.py`, `debug_cuda_error.py`, etc. These additional scripts were created to ensure experimental stability and reproducibility due to the stochastic nature of GAN convergence.

---

### 2. Domain Adaptation Experiments

1. Navigate to the `Domain Adaptation` folder.

2. Download the dataset
    - For ImageNetR dataset: Navigate to the `data\ImageNetR` folder and run `bash get_iamgenet.sh`. If failed, try `bash download_iamgenet_r.sh`. Download and unzip the [image_list](`https://cloud.tsinghua.edu.cn/f/7786eabd3565409c8c33/?dl=1`) here as well.
    - For Office31 dataset: Navigaet to the `data\Office31` folder and run `bash get_office31.sh` and unzip `image_list.zip`.
    - For OfficeHome dataset: Navigaet to the `data\OfficeHome` folder and run `bash get_officehome.sh` and unzip `image_list.zip`.

3. Run the following scripts:
    - For SCA+MCC method 
        - with CLIP embedding input of Office31 dataset for all Source-Target combinations:
            ```bash 
            bash run_train_all.sh --gpus 0,1 --concurrent 1 --model "clip ViT-L/14 768" --data_name Office31
            ```
        - with ResNet50 embeddings input of ImageNetR dataset for specific Source-Target adaptation:
            ```bash 
            bash run_train_exp.sh --source IN-val --source INR --model "resnet resnet50 2045" --data_name ImageNetR --gpu 2
            ```
    - For SCA method:
        ```bash
        bash run_train_nomcc_all.sh
        ```
        For more parameter settings, please refer to the parameter descriptions in the script files, such as the command line parameter options in `run_train_exp.sh` and `run_train_all.sh` or `run_train_nomcc_all.sh` and `run_train_nomcc_exp.sh`
    - For DANN method:
        ```bash
        bash run_dann_all.sh
        ```
    - For MDD method:
        ```bash
        bash run_mdd_all.sh
        ```
    - For MCC method:
        ```bash
        bash run_mcc_all.sh
        ```
    For more parameter settings or running individual experiments, you can use the corresponding experiment scripts, for example:
    ```bash
    bash run_dann_exp.sh --source A --target W --model "clip ViT-L/14 768" --data_name Office31 --gpu 0
    ```
    Each method has its corresponding experiment script:
    - `run_dann_exp.sh`: Single experiment for DANN method
    - `run_mdd_exp.sh`: Single experiment for MDD method  
    - `run_mcc_exp.sh`: Single experiment for MCC method
    - `run_train_nomcc_exp.sh`: Single experiment for SCA method
    - `run_train_exp.sh`: Single experiment for SCA+MCC method
#### Newly developed code

All shell script files (`.sh`) in this directory are **newly developed** for this project to facilitate the execution of various domain adaptation experiments with different methods and parameter configurations.

The core Python implementation files `dann.py`, `mcc.py`, and `mdd.py` are **adapted from** the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library). Specifically, we modified the data loader components in the original source code to replace the input features with embeddings, making them consistent with our SCA method's input format. This modification enables fair comparison between our SCA approach and existing baseline methods using the same feature representations.



### 3. Word Embedding Alignment

1) Navigate to the `Word Alignment` folder and create a `data` directory:
   ```bash
   mkdir data
   cd data/
    ```
2) Download and extract pre-trained word vectors inside `data`:

    ```bash
    wget https://dl.fbaipublicfiles.com/arrival/vectors.tar.gz
    tar -xvf vectors.tar.gz
    ```

3) Prepare cross-lingual data:
    - Inside `data/`, create a `crosslingual` directory:
    ```bash
    mkdir crosslingual
    cd crosslingual
    ```

    - Download and extract the required files inside `crosslingual`:
    ```bash
    wget https://dl.fbaipublicfiles.com/arrival/wordsim.tar.gz
    wget https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
    tar -xvf wordsim.tar.gz
    tar -xvf dictionaries.tar.gz
    ```

4) Run the alignment script:
    ```bash
    bash run.sh
    ```


### 4. Single-Cell Sequence Analysis
This experiment uses utilities (`Single cell sequence analysis/single_cell/`) adapted from the [Cross-modal autoencoders](https://github.com/uhlerlab/cross-modal-autoencoders). Follow these steps:

1) Navigate to the `Single cell sequence analysis` folder.
2) Run the script:
```bash
bash run.sh
```

#### Newly developed code

The following scripts and utilities are newly developed for this project:

- `run_multiple_seeds.sh`: A shell script for running experiments with multiple random seeds to ensure reproducibility and statistical significance.
- `test_different_D.sh`: A shell script for testing different values of the shared component dimension D.
- Visualization utilities: Python scripts for generating plots and figures to analyze and present experimental results.
