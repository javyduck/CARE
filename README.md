# CARE: Certifiably Robust Learning with Reasoning Pipeline

## Introduction

Deep Neural Networks (DNNs) have revolutionized a multitude of machine learning applications but are notorious for their susceptibility to adversarial attacks. Our project introduces [**CARE (Certifiably Robust leArning with REasoning)**](https://arxiv.org/abs/2209.05055), aiming to enhance the robustness of DNNs by integrating them with reasoning abilities. This pipeline consists of two primary components:

- **Learning Component**: Utilizes standard DNNs for semantic predictions, e.g., recognizing if an input image contains something furry.
- **Reasoning Component**: Employs probabilistic graphical models like Markov Logic Networks (MLN) to apply domain-specific knowledge and logic reasoning to the learning process.

## Repository Contents

- `code/`: Contains the source code for our experiments.
- `data/`: Folder where the datasets will be downloaded.

## Getting Started

### Prerequisites

`conda create -n name care python=3.8`

`conda activate care && pip install -r requirements.txt`

### Installation and Running

1. **Train the Main Sensor:**

   ```
   bashCopy code
   python code/train_main.py --dataset AWA --arch resnet50 --outdir saved_models/AWA/main_models/noise_sd_0.25/ --noise_sd 0.25 --epochs 30 --batch 256 --lr 0.001 --lr_step_size 20
   ```

2. **Train Knowledge Predictors (Attribute and Hierarchy Predictors):**

   ```
   bashCopy code
   python code/train_attribute.py --dataset AWA --arch resnet50 --outdir saved_models/AWA/attribubte_models/noise_sd_0.25/ --noise_sd 0.25 --epochs 30 --batch 256 --lr 0.001 --lr_step_size 20 --weight 2
   
   python code/train_hierarchy.py --dataset AWA --arch resnet50 --outdir saved_models/AWA/hierarchy_models/noise_sd_0.25/ --noise_sd 0.25 --epochs 30 --batch 256 --lr 0.001 --lr_step_size 20 --weight 2
   ```

3. **Train the GCN:**

   ```
   bashCopy code
   python code/train_gcn.py --dataset AWA --noise_sd 0.25 --w 0.5
   ```

4. **Certification:**

   ```
   bashCopy code
   python code/certify.py AWA 0.25 --outdir prediction_results/AWA/noise_0.25 --alpha 0.001 --N0 100 --N 10000
   ```

### Architecture

We define the base classifier in `code/core.py` to process input images through a sequence of operations. Initially, the images pass through the main model and various knowledge models to obtain corresponding confidences. These confidences are concatenated into a single vector, which is then fed into the Graph Convolutional Network (GCN) for the final prediction.

## Citation & Further Reading

If you find our work useful, please consider citing our [paper](https://arxiv.org/abs/2209.05055):

```
@inproceedings{zhang2023care,
  title={CARE: Certifiably Robust Learning with Reasoning via Variational Inference},
  author={Zhang, Jiawei and Li, Linyi and Zhang, Ce and Li, Bo},
  booktitle={2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  pages={554--574},
  year={2023},
  organization={IEEE}
}
```

## Upcoming Features

We are actively working on integrating CLIP as the knowledge sensor to further optimize our pipeline. This new feature is aimed at enhancing the robustness and scalability of the model. Stay tuned for this exciting update, which will be released soon!

## Contact

If you have any questions or encounter any errors while running the code, feel free to contact [jiaweiz@illinois.edu](mailto:jiaweiz@illinois.edu)!