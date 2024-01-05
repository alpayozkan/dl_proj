# Deep Learning Project

Welcome to our Deep Learning Project repository.

In this project, we explored four different topics to compare Multi-Layer Perceptrons (MLPs) with vision models such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs):

- Adversarial Attacks
- Class Maximization
- Saliency Maps
- Dimensionality Reduction

For instructions on running the code related to each topic, please refer to the information provided in the corresponding section below.

The required model weights can be found on this [Drive](https://drive.google.com/drive/folders/11Ma2KYE_OuFOHYrvX1R5ZWZtWJ4gC6c4) under the subfolder finetuned_models.

## Adversarial Attacks 

To run the adversarial attacks code for our experiments, follow these steps:

1. Switch to the `adv_attacks` branch:

    ```bash
    git checkout adv_attacks
    ```

2. Inside the `dl_proj` folder:

    - Download the `adversarial-attacks-pytorch` folder from [this repository](https://github.com/Harry24k/adversarial-attacks-pytorch.git).

    - Create the following folders and add the corresponding weights from the Drive:

        - `state_dicts` folder:
            - `resnet18.pt`

        - `finetuned_models` folder:
            - `vit_tiny_patch16_224_unfrozen_CIFAR.t7`

        - `checkpoints` folder:
            - `mlp_b12_wi1024_imagenet_bs128_tinyimagenet.t7`
            - `ResNet18_TinyImageNet.t7`
            - `vit_tiny_patch16_224_unfrozen_tinyimagenet.t7`

3. To run the experiments on CIFAR10, execute the cells in the notebook named `CIFAR10_AdversarialAttacks.ipynb`.

4. To run the experiments on Tiny ImageNet, execute the cells in the notebook named `TinyImageNet_AdversarialAttacks.ipynb`.


## Class Maximization

To run the class maximization code, follow these steps:

1. Switch to the `classmaxim` branch:

    ```bash
    git checkout classmaxim
    ```

2. Inside the `dl_proj` folder:

    - Create the following folder and add the corresponding weights from the Drive:

        - `checkpoints` folder:
            - `mlp_b12_wi1024_imagenet_bs128_tinyimagenet.t7`
            - `ResNet18_TinyImageNet.t7`
            - `vit_tiny_patch16_224_unfrozen_tinyimagenet.t7`

3. To run the class maximization code, open the `revelio` folder and execute the cells in the notebook named `cm-tinyimgnet.ipynb`.


## Saliency
### Occlusion Sensitivity
We ran occlusion sensitivity on Google Colab. The jupyter notebook can be found under the shared Google [Drive](https://drive.google.com/drive/folders/11Ma2KYE_OuFOHYrvX1R5ZWZtWJ4gC6c4) under the name "sensitivity_occlusion_final.ipynb".
To run this file, add a shortcut to the shared Google Drive folder to your Google Drive. Then open the juypter notebook with Google Colab and adjust the PATH in the second cell.
To see the result files, head to the shared Google Drive folder and look for the subfolder: occlusion_sensivity_tinyimagenet_true_class. Each image is titled with the TinyImageNet Validation ID for that reference image.

### LIME and Dimensionality Reduction
1. Switch to the `saliency` branch:

    ```bash
    git checkout saliency
    ```

2. Inside the `dl_proj` folder:

    - Inside the following folder, add the corresponding weights from the Drive:

        - store to `models`:
            - `mlp_b12_wi1024_imagenet_bs128_tinyimagenet.t7`
            - `ResNet18_TinyImageNet.t7`
            - `vit_tiny_patch16_224_unfrozen_tinyimagenet.t7`
        
    - rename the downloaded files:
      - `mlp_b12_wi1024_imagenet_bs128_tinyimagenet.t7` -> `mlp_tinyimagenet.pt`
      - `ResNet18_TinyImageNet.t7` -> `resnet18_tinyimagenet.pt`
      - `vit_tiny_patch16_224_unfrozen_tinyimagenet.t7` -> `ViT_tinyimagenet.pt`

3. There are 4 important notebooks:
    - lime_tester.ipynb
    - lime_experiments.ipynb
    - dimensionality_tester.ipynb
    - dimensionality_experiments.ipynb
    
    The first two are related to LIME. The 'tester' notebook consists of some basic applications of LIME.
    The second notebook can be used to reproduce our experimental results. More details can be found in the notebooks.
    
    Similar for dimensionality reduction, we are introduced to the topic through the 'tester' notebook, whereas the 
    experiments can be performed by executing the second notebook.
