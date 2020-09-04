import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

from parameters import *
from networks.nets import *
from eval.evaluator import *

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    branch_num = 1
    epoch_num = 40
    # ========================================== dir & param ==========================================
    data_dir = r'/home/user/Desktop/test_dataset'
    weight_dir = os.path.join('./save', 'branch_'+str(branch_num), model_name+'_epoch_'+str(epoch_num)+'.pth')
    dst_dir = os.path.join(data_dir, 'predicton/branch_1')
    # store_num = 10
    the_name = os.path.splitext(os.path.basename(weight_dir))[0]
    # assert test_params['test_batch'] >= store_num, 'batch size must be bigger than the number of storing image.'
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
    test_set = ImageFolder(root='./data/test', transform=transform_set)
    label_name = test_set.classes
    print(f'test set classes : {label_name}')
    print(f'test data : {len(test_set)} files detected.')
    test_loader = DataLoader(dataset=test_set, batch_size=test_params['test_batch'],
                              shuffle=False, num_workers=user_setting['test_processes'])
    table = {i: label for i, label in enumerate(label_name)}
    evaluator = Evaluator(class_table=table)
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== Model Load ==========================================
    netend = Classifier(2)
    model = ResNet50(netend, 2)
    print(f'target weight: {weight_dir}')
    model.load_state_dict(torch.load(weight_dir))
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
    os.makedirs(dst_dir, exist_ok=True)
    print(f'save directory : {dst_dir}')
    # ============================================ run ================================================
    model.eval()
    for i, data in enumerate(test_loader):
        image, label = data

        if environment['gpu']:
            image = image.cuda()

        with torch.no_grad():
            output = model.forward(image)

        value, indices = output.max(-1)

        if environment['gpu']:
            value = value.cpu()
            indices = indices.cpu()
            label = label.cpu()

        # print(f'pred: {indices}')
        # print(f'label: {label}')

        evaluator.record(indices, label)
        print(f'{(i+1) / len(test_loader) * 100:.2f} % processed.')
    # =================================================================================================
    print(f'Accuracy: {evaluator.accuracy():.3f}')
    print(f'Precision: {evaluator.precision():.3f}')
    print(f'Recall: {evaluator.recall():.3f}')
    print(f'F1 Score: {evaluator.f1_score():.3f}')
    evaluator.heat_map()
    # ------------------------------------------------------------------------------------------------------------------

