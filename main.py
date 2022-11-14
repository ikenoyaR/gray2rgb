import argparse

from train import train

def main(args):
    if args.mode == 'train':
        pass
    elif args.mode =='test':
        pass
    elif args.mode == 'predict':
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--img-size', type=str, default='640x640', help='inference size (pixels) "height x width"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/RGB_image', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--mode', default='test', choices=['train', 'test', 'predict', 'debug'], help='choose modes train or test or predict')
    args = parser.parse_args()
    print(args)
    main(args)