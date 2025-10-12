


class DefaultConfig(object):
    scaling_factor = 4

    data_root = './data/train_data/X%d/BTM_LR/'%(scaling_factor)
    HR_root = './data/train_data/X%d/BTM_HR/'%(scaling_factor)
    label_root='./data/train_data/X%d/BTM_label/'%(scaling_factor)


    num_data = 4000

    batch_size = 32
    use_gpu = True
    num_workers = 0

    max_epoch = 1
    lr = 0.0005
    lr_decay = 0.5

    load_model_path = './model/BTM_SISR%d.pth' % (scaling_factor)
    save_model_path = './model/BTM_SISR%d.pth' % (scaling_factor)

opt = DefaultConfig()

























