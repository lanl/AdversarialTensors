# AdversarialTensors: Tensors-based framework for adversarial robustness

This library implements a variety of tensor factorization methods for defending Artificeal intelligence 
(AI) models against adversarial attacks. The library implements three main operations. First, tensor
factorization methods are implemented as a preprocessing stage for input data to AI models to reduce the
effectiveness of adversarial noise.  In the second operation, tensor factorization methods are used to 
find novel latent attack features by combining proposed attacks from a variety of methods. Since these 
attacks will inherently be a combination of attacks many algorithms against many models, they have the 
potential to threaten a wide variety of AI models simultanioulsy In the third operation, an unsupervised
generative adversarial networks (GAN) is employed to  generate denoised data from many adversarial noises. 
This generator provides robust defense against unseen attacks. 

## Capabilites:
* Tensor factorizations with Tucker, non-negative Tucker, CPD, non-negative CPD, Tensor Train, NMF, 
NNSVD and NMF

<p align="center">
  <img width="500" height="322" src="docs/denoising.png">
</p>
* Ability to denoise image single/batches/all.
<p align="center">
  <img width="500" height="156" src="docs/inference.png">
</p>
* Employs attacks such as AutoAttack, FGSM, PGD, BIM, DeepFool. 
* Ability to perform white-box/black-box attacks.

<p float="center">
  <img width="400" height="270" src="docs/white_box.png">
  <img width="400" height="211" src="docs/black_box.png">
</p>

* Ability to perform adversarial training.

<p float="center">
  <img width="400" height="270" src="docs/white_box_adv_train.png">
  <img width="400" height="211" src="docs/black_box_adv_train.png">
</p>

* Works on Cifar10/Cifar100/Imagenet/Tiny-Imagenet/MNIST/Fashion-MNIST. 



# End-to-End Code Execution Guide

## **1. Training Models**

First, set the environment variables specific to your experiment in `send_train_jobs.sh`.

### **Output:**

- Models will be stored in `my_checkpoints`. For example: `my_checkpoints/cifar10/resnet18_0_1.ckpt`
- If using CSVLogger, log data will be saved in `log_{dataset_name}_{model_name}_test/{fold_ind}_{nfolds}`

### **To Run:**

\`\`\`bash
bash send_train_jobs.sh
\`\`\`

## **2. Generating Attack Data**

Set the environment variables for the experiment in `send_attack_jobs.sh`.

### **Output:**

- Attack data and logs will be stored in `attack_log_{dataset_name}/{model_name}/{fold_ind}_{nfolds}_{attack_type}`

### **To Run:**

\`\`\`bash
bash send_attack_jobs.sh
\`\`\`

## **3. Parameter Tuning**

Set the environment variables for the experiment in `send_tune_jobs.sh`.

### **Output:**

- The tuning results will be located at `parameter_search_{dataset_name}/{model_name}/{fold_ind}_{nfolds}_{attack_type}/search_results`

### **To Run:**

\`\`\`bash
bash send_tune_jobs.sh
\`\`\`

## **4. Collecting Results**

After setting the environment variables:

### **Output:**

- The results will be stored as CSV files in `tune_csv_results`

### **To Run:**

\`\`\`bash
bash collect_results.sh
\`\`\`

## **5. Final Evaluation on Test Set**

### **Generating Attacks**

- Set `final_test_set` to `True` in `send_attack_jobs.sh`

\`\`\`bash
bash send_attack_jobs.sh
\`\`\`

### **Evaluating Models**

- After setting specific environment variables (such as `eval_top_k`):

### **To Run:**

\`\`\`bash
bash send_final_eval_jobs.sh
\`\`\`
