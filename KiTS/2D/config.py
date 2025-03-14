import argparse

def GenConfig():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='/mai_nas/LSH/Data/KiTS_2D/', help = '')
    parser.add_argument('--restart', type=str, default='')
    parser.add_argument('--model_save_dir', type=str, default='KiTS/Model_Save/', help='')
    
    parser.add_argument('--GPU', type=str, default='2')
    parser.add_argument('--dataset', type=str, default='KiTS', help='')
    parser.add_argument('--model_name', type=str, default='UNet', help='')
    parser.add_argument('--loss_function', type=str, default='DICE+AC+FOCAL', help='')
    
    
    parser.add_argument('--clip_min', type=int, default=-150)
    parser.add_argument('--clip_max', type=int, default=800)
    
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--val_batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate') # start: 0.00005
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--sample_interval', type=int, default=10, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
    parser.add_argument('--data_augment', type=bool, default=False, help='32-fold data augmentation')
    
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--eta', type=float, default=0.8)
    
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=4)
    parser.add_argument('--num_features', type=int, default=32)

    opt = parser.parse_args()
    print(opt)
    
    return opt
