
## Object recognition and computer vision 2019/2020

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA to enable GPU acceleration

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

On Linux/Mac, the dataset can be downloaded by executing
```
./download_dataset.sh
```

#### Training and validating your model

Run the script `main.py` to train your model.

##### Noisy Student

To increase the accuracy of the predicted labels, you can use a method called **Noisy Student**.
When running `main.py`, add option `--student N` to add the less uncertain unlabeled images to the training set.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Results

Table below details the effect of improvements on the overall accuracy:

| Improvement                    | Accuracy (%) |
|--------------------------------|--------------|
| None                           | 61.9         |
| Data augmentation              | 69.0         |
| DA + LR scheduler              | 77.4         |
| **DA + scheduler + Noisy Student** | **78.1** |

#### References

```
@article{NoisyStudent, 
arxivId = {1911.04252},
author = {Xie, Qizhe and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V.},
month = {nov},
title = {{Self-training with Noisy Student improves ImageNet classification}},
url = {http://arxiv.org/abs/1911.04252},
year = {2019}
}

@article{ResNext101,
arxivId = {1905.00546},
author = {Yalniz, I. Zeki and J{\'{e}}gou, Herv{\'{e}} and Chen, Kan and Paluri, Manohar and Mahajan, Dhruv},
month = {may},
title = {{Billion-scale semi-supervised learning for image classification}},
url = {http://arxiv.org/abs/1905.00546},
year = {2019}
}

```

#### Acknowledgments

Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
