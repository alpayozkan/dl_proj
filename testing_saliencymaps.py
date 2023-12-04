import random
import os
import torch
import matplotlib

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.networks import get_model
from data_utils.data_stats import *
from utils.metrics import AverageMeter, real_acc
from torchmetrics import Accuracy

from models.resnet import resnet18
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

matplotlib.use('TkAgg')  # workaround, otherwise I get error


def setup():
    print("Setting up dataset and models...")

    # Dataset configuration
    dataset_name = 'cifar10'  # One of cifar10, cifar100, stl10, imagenet or imagenet21
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    mean = MEAN_DICT[dataset_name]
    std = STD_DICT[dataset_name]
    num_classes = CLASS_DICT[dataset_name]
    dataset_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    assert len(dataset_classes) == num_classes

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=512,
                                              shuffle=False,
                                              num_workers=2)

    print("dataloader initialized.")

    # MLP Configuration
    mlp_architecture = 'B_12-Wi_512'  # B_12-Wi_512
    mlp_resolution = 64  # Resolution of fine-tuned model (64 for all models we provide)
    mlp_transform = transforms.Compose([
        transforms.Resize((mlp_resolution, mlp_resolution)),
        transforms.Normalize(mean / 255., std / 255.)
    ])
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    mlp_checkpoint = 'in21k_cifar10'  # This means you want the network pre-trained on ImageNet21k and finetuned on CIFAR10
    mlp_model = get_model(architecture=mlp_architecture, resolution=mlp_resolution, num_classes=num_classes,
                          checkpoint=mlp_checkpoint)

    print("MLP model loaded")

    # Resnet Configuration
    resnet_model = resnet18(pretrained=True)
    resnet_transform = transforms.Compose([
        transforms.Normalize(mean / 255., std / 255.)
    ])

    target_layers = [resnet_model.layer4[-1]]
    create_saliency_maps(resnet_model, resnet_transform, target_layers, img_per_class=2, reshape=False)
    # print(mlp_model.BottleneckBlock[11])
    # target_layers = [resnet_model.layer4[-1]]
    # create_saliency_maps(mlp_model, resnet_transform, target_layers, img_per_class=2, reshape=False)

    print("ResNet model loaded.")

    return {"mlp": (mlp_model, mlp_transform),
            # "resnet": (resnet_model, resnet_transform)
            }, data_loader


def evaluate_model(model: torch.nn.Module, data_loader, transform, reshape=False) -> float:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_acc = AverageMeter()
    accuracy = Accuracy(task="multiclass", num_classes=10)

    with torch.no_grad():
        for ims, targs in data_loader:
            ims = transform(ims)
            if reshape:
                ims = torch.reshape(ims, (ims.shape[0], -1))
            preds = model(ims)
            probabilities = torch.nn.functional.softmax(preds, dim=0)
            predicted_class = torch.argmax(probabilities, dim=-1)
            acc = accuracy(predicted_class, targs)
            total_acc.update(acc, ims.shape[0])

    return total_acc.get_avg()


def evaluate_models():
    torch.backends.cuda.matmul.allow_tf32 = True

    model_configs, loader = setup()

    for name, (model, transform) in model_configs.items():
        print(f"Evaluating {name}...")
        acc = evaluate_model(model, loader, transform, name == "mlp")
        print(f"Accuracy for {name}: {acc}")


def create_saliency_maps(model, transform, target_layers, img_per_class=1, reshape=False) -> None:
    dataset_name = 'cifar10'
    plain_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    plain_loader = torch.utils.data.DataLoader(plain_dataset,
                                               batch_size=512,
                                               shuffle=True,
                                               num_workers=2)
    num_classes = CLASS_DICT[dataset_name]
    dataset_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    output_trf = transforms.Resize((128, 128))
    images = [[] for _ in range(num_classes)]

    for ims, targs in plain_loader:
        for image, y_true in tuple(zip(ims, targs)):
            if len(images[y_true.item()]) < img_per_class:
                images[y_true.item()].append(image)

        for arr in images:
            if len(arr) < img_per_class:
                continue
        break

    for class_index in range(len(images)):
        for img_index in range(len(images[class_index])):
            img = output_trf(images[class_index][img_index]).numpy().transpose((1, 2, 0))
            plt.imsave(f"outputs/{dataset_classes[class_index]}_{img_index}.png", img)

    for class_index in range(len(images)):
        for img_index in range(len(images[class_index])):
            img = images[class_index][img_index]

            targets = [ClassifierOutputTarget(class_index)]
            # GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
            with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
                trf_img = transform(img)
                if reshape:
                    trf_img = torch.reshape(trf_img, (trf_img.shape[0], -1))
                grayscale_cam = cam(input_tensor=trf_img.unsqueeze(0), targets=targets, aug_smooth=True, eigen_smooth=True)

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]

                visualization = show_cam_on_image(img.numpy().transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
                visualization = output_trf(torch.from_numpy(visualization.transpose(2, 0, 1))).numpy().transpose((1, 2, 0))
                plt.imsave(f"outputs/{dataset_classes[class_index]}_{img_index}_saliency.png", visualization)


def main():
    dataset = 'cifar10'
    mean = MEAN_DICT[dataset]
    std = STD_DICT[dataset]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    image_nr = random.randrange(len(cifar10_dataset))
    image, true_label = cifar10_dataset[image_nr]

    model = resnet18()
    target_layers = [model.layer4[-1]]

    targets = [ClassifierOutputTarget(true_label)]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        model_image = transforms.Normalize(mean / 255., std / 255.)(image).unsqueeze(0)
        grayscale_cam = cam(input_tensor=model_image, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(image.numpy().transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.show()


if __name__ == "__main__":
    # evaluate_models()
    setup()
