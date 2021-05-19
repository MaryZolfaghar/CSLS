# Complementary Structure Learning Systems

This repository contains the code for [this paper](https://arxiv.org/).

**Title:** Complementary Structure-Learning Neural Networks for Relational Reasoning

**Abstract:** The neural mechanisms supporting flexible relational inferences, especially in novel situations, are a major focus of current research. In the complementary learning systems framework, pattern separation in the hippocampus allows rapid learning in novel environments, while slower learning in neocortex accumulates small weight changes to extract systematic structure from well-learned environments. In this work, we adapt this framework to a task from a recent fMRI experiment where novel transitive inferences must be made according to implicit relational structure. We show that computational models capturing the basic cognitive properties of these two systems can explain relational transitive inferences in both familiar and novel environments, and reproduce key phenomena observed in the fMRI experiment.

**Keywords:**
neural networks; cognitive maps; complementary learning systems; structure learning; transitive inference

To cite this paper:
```
@inproceedings{
  RussinZolfagharParkEtAl21,
  title = {Complementary {{Structure}}-{{Learning Neural Networks}} for {{Relational Reasoning}}},
  booktitle = {Proceedings for the 43nd {{Annual Meeting}} of the {{Cognitive Science Society}}},
  author = {Russin, J. L. and Zolfaghar, M. and Park, S. A. and Boorman, E. and O'Reilly, R. C.},
  year = {2021},
  pages = {7},
}
```



## Scripts Info.
This directory containts the following:
- [`main.py`](main.py): Runs train, test and analyze for both the episodic and the cortical system
- [`dataset.py`](dataset.py): Classes for generating and building the dataset and corresponding dataloader
- [`models.py`](models.py): Models used in both the episodic and the cortical system
- [`train.py`](train.py): Training script with periodic validation 
- [`test.py`](test.py): Testing script 
- [`results.ipynb`](results.ipynb) Reproducing the results in the paper.
- [`results.P`](results.P) Contains results 

<!-- ## Installation
### Conda (Recommended)

If you are using conda, you can create a `csls` environment with all the dependencies by running: -->

<!-- ```
git clone https://github.com/MaryZolfaghar/ComplementaryStructureLearningSystems
cd ComplementaryStructureLearningSystems
conda env create -f environment.yaml
source activate csls
```


(ToDo: check if this is necessary)
Then, execute the following command to installs the repository in editable mode.

```
pip install --editable .
``` -->


### Manual Installation
```
conda create -n csls python=3.6
conda activate csls
conda install pytorch torchvision torchaudio -c pytorch
conda install -c anaconda scikit-learn 
```
Note that you should install [PyTorch](http://pytorch.org/) for on your platform.

## Usage
To train and test both models with the default parameters and using the face images:
```bash
python main.py --use_images
```
The default hyperparameters and their descriptions can be found in `main.py`.

```
usage: main.py [--use_cuda False][--seed 0]
               [--print_every 200][--lr_cortical 0.001]
               [--out_file results.P][--N_episodic 1000]
               [--bs_episodic 16][--lr_episodic 0.001]
               [--use_images False][--image_dir images/]
               [--N_cortical 1000][--bs_cortical 32]               
```
