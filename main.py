import argparse

from train import train

def main(args):
    args.img_size = tuple(map(int, args.img_size.split('x')))
    if args.mode == 'train':
        train(args)
    elif args.mode =='test':
        pass
    elif args.mode == 'predict':
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/ImageNet', help='Path of dataset')   
    parser.add_argument('--weights', nargs='+', type=str, default='model.pt', help='model.pt path(s)')

    # input
    parser.add_argument('--img_size', type=str, default='224x224', help='inference size (pixels) "height x width"')
    parser.add_argument("--batch_size", type=int, default=16, help="number of batch_size")
    
    # training parameter
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs you want to train")
    parser.add_argument("--workers", type=int, default=4, help="number of threads")
    parser.add_argument("--pin_memory", action='store_false', help="argument of PyTorch Dataloader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--scheduler", type=str, default=None, help="Learning rate sheduler")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/RGB_image', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--mode', default='test', choices=['train', 'test', 'predict', 'debug'], help='choose modes train or test or predict')
    args = parser.parse_args()
    print(args)
    main(args)