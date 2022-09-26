# Code for KiU-Net_Paddle
# #######################
import argparse
import importlib
import os


def argparsing():
    parser = argparse.ArgumentParser(description='KiU-Net_Paddle')
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH', help='path to root path of lfw dataset (default: ../lfw)')
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--modelname', default='off', type=str, help='turn on img augmentation (default: False)')
    parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
    parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
    parser.add_argument('--load', default='default', type=str, help='turn on img augmentation (default: default)')
    parser.add_argument('--save', default='default', type=str, help='turn on img augmentation (default: default)')
    parser.add_argument('--model', default='kiunet', type=str, help='model name')
    parser.add_argument('--direc', default='./brainus_OC_udenet', type=str, help='directory to save')
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--edgeloss', default='off', type=str)
    parser.add_argument('--prediction_only', action='store_true', help='prediction?')
    parser.add_argument('--backend', default='paddle', choices=['keras', 'pytorch', 'paddle'], type=str, help='which backend to use?')
    parser.add_argument('--model_path', default=None, type=str, help='path to model check')

    args = parser.parse_args()

    print()
    print('============================================================')
    print(args)
    print('============================================================')
    print()

    return args


def main(args, CORE):
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            print('Model path has been verified.')
        else:
            print('Invalid model path! Please specify a valid model file. Program terminating...')
            exit()
    if args.prediction_only:
        CORE.test(args)
        exit()
    CORE.train(args)
    CORE.evaluate(args)


if __name__ == "__main__":
    args = argparsing()
    CORE = importlib.import_module(args.backend + '_version')
    main(args, CORE)
