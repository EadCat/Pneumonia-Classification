import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from parameters import *
from networks.nets import *
from eval.evaluator import *
from dataloader.dataset import DictFolder
from functions.utils import *
from path import DataManager, DirectoryManager

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # ========================================== dir & param ==========================================
    mode = 'test'
    branch_num = 6 # <-- modify here (branch directory)
    epoch_num = 49 # <-- modify here (weights epoch number)
    dir_man = DirectoryManager(model_name=model_name, mode=mode, branch_num=branch_num,
                               load_num=epoch_num)
    data_man = DataManager(os.getcwd())
    # assert test_params['test_batch'] >= store_num, 'batch size must be bigger than the number of storing image.'
    # =================================================================================================

    # =========================================== Model Load ==========================================
    netend = Classifier1(2)  # <- model definition
    model = ResNet18(netend, pretrain=permission['pretrain'])  # <-- model definition
    print(f'target weight: {dir_man.load()}')
    model.load_state_dict(torch.load(dir_man.load()))
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== transform ===========================================
    transform_set = transforms.Compose([
        transforms.Resize(size=test_params['resized']),
        transforms.ToTensor(),
        transforms.Normalize(mean=test_params['mean'], std=test_params['std'])
    ])
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ============================================ Dataset ============================================
    test_set = DictFolder(root='./data/test', transform=transform_set)
    label_name = test_set.classes
    print(f'test set classes : {label_name}')
    print(f'test data : {len(test_set)} files detected.')
    test_loader = DataLoader(dataset=test_set, batch_size=test_params['test_batch'],
                              shuffle=False, num_workers=user_setting['test_processes'])
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== Evaluator ===========================================
    table = {i: label for i, label in enumerate(label_name)}
    evaluator = Evaluator(class_table=table)
    # =================================================================================================

    # ======================================= image Sampler set =======================================
    # Image Captioning Instances
    # Model Prediction Captioning
    positive_cap = Captioner('Pneumonia', (225, 750), color=(0, 0, 255))
    negative_cap = Captioner('Normal', (275, 750), color=(255, 0, 0))
    # Ground Truth Captioning
    gt_negative = Captioner('GT: Normal', (225, 100), color = (230, 230, 230))
    gt_positive = Captioner('GT: Pneumonia', (165, 100), color=(255, 255, 255))
    # number of images to be saved per batch.
    store_num = 1
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ========================================== GPU setting ==========================================
    environment = {}
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU {torch.cuda.get_device_name()} available.')
        model.cuda()
        environment['gpu'] = True

    else:
        device = torch.device('cpu')
        print(f'GPU unable.')
        environment['gpu'] = False
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ============================================ run ================================================
    model.eval()
    for i, data in enumerate(test_loader):
        image, label, name = data[tag_image], data[tag_label], data[tag_name]

        if environment['gpu']:
            image = image.cuda()

        with torch.no_grad():
            output = model.forward(image)

        value, indices = output.max(-1)

        # VRAM -> RAM copy
        if environment['gpu']:
            value = value.cpu()
            indices = indices.cpu()
            label = label.cpu()
            image_for_sample = image[:store_num].cpu()
        else:
            image_for_sample = image[:store_num]

        # image labeling and sample
        image_for_sample = dimension_change(image_for_sample)
        image_for_sample = np.ascontiguousarray(image_for_sample, dtype=np.uint8)

        # image captioning
        for j, (img, gt, title, index) in enumerate(zip(image_for_sample, label, name, indices)):
            if gt == 0:  # normal
                gt_negative.write(img)
            elif gt == 1:  # pneumonia
                gt_positive.write(img)

            if index == 0: # normal
                negative_cap.write(img)
            elif index == 1: # pneumonia
                positive_cap.write(img)

        # sample image store
        imgstore(image_for_sample*100.0, store_num, save_dir=dir_man.test_sample(), epoch=epoch_num, cls='test', filename=name)

        # record for evaluation
        evaluator.record(indices, label)

        print(f'{(i+1) / len(test_loader) * 100:.2f} % processed.')
    # =================================================================================================
    print(f'Accuracy: {evaluator.accuracy():.3f}')
    print(f'Precision: {evaluator.precision():.3f}')
    print(f'Recall: {evaluator.recall():.3f}')
    print(f'F1 Score: {evaluator.f1_score():.3f}')
    write_line({'Precision': evaluator.precision()}, os.path.join(dir_man.test(), 'model eval.txt'))
    write_line({'Recall': evaluator.recall()}, os.path.join(dir_man.test(), 'model eval.txt'))
    write_line({'Accuracy': evaluator.accuracy()}, os.path.join(dir_man.test(), 'model eval.txt'))
    write_line({'F1 Score': evaluator.f1_score()}, os.path.join(dir_man.test(), 'model eval.txt'))
    evaluator.heat_map(store_dir=os.path.join(dir_man.test(), 'heatmap.png'))
    # ------------------------------------------------------------------------------------------------------------------

