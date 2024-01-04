# Deep Learning Project

Welcome to our Deep Learning project repository.

In this project, we explored four different topics to compare Multi-Layer Perceptrons (MLPs) with vision models such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs):

- Adversarial Attacks
- Class Maximization
- Saliency Maps
- Dimensionality Reduction

For instructions on running the code related to each topic, please refer to the information provided in the corresponding section below.

The required model weights can be found on this [Drive](https://drive.google.com/drive/folders/11Ma2KYE_OuFOHYrvX1R5ZWZtWJ4gC6c4).

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

3. To run the class maximization code, navigate to the `revelio` folder and execute the cells in the notebook named `cm-tinyimgnet.ipynb`.


## Saliency
### Occlusion Sensitivity
TODO
### LIME
TODO

## Dimensionality Reduction
TODO