
import os
import torch
from lib.dataset.landsat_8_dataloader import Get_dataloader
from lib.model.create_generator import create_generator
from lib.model.generator.ms_net import MS_Net
from lib.model.generator.msa_net import MSA_Net
from lib.model.generator.rsc_net import RSC_Net
from lib.model.generator.ssa_net import SSA_Net
from lib.utils.config import Config
from lib.utils.parse_args import parse_args
from rm_cloud.single_model_methods.tester import Testner


def eval():
    config = Config().parse()
    args = parse_args(config)
    print(args)
    train_loader, test_loader = Get_dataloader(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    if args.model.arch == 'rsc':
        model = RSC_Net()
    if args.model.arch == 'msa':
        model = MSA_Net()
    if args.model.arch == 'ssa':
        model = SSA_Net()

    if args.model.arch == 'ms':
        model = MS_Net()

    if args.model.arch == 'msa_':
        model = MSA_Net()
    all_eval_dir = '/home/data1/xiesong/git_repo/removal_thin_cloud/experiments/'+ args.model.arch + '/every_model/'

    list_checkpoint = os.listdir(args.checkpoint_dir)
    for checkpoint in list_checkpoint:
        if int(checkpoint.split('-')[0]) > 3018:
            checkpoint_path = os.path.join(args.checkpoint_dir + checkpoint)
            args.evaluation_dir = os.path.join(all_eval_dir + checkpoint.split('-')[0] + '/')
            os.makedirs(args.evaluation_dir,exist_ok=True)
            print(args.evaluation_dir)
            c_model = create_generator(args, model)

            c_model.load_state_dict(torch.load(checkpoint_path))
            test_model = Testner(c_model)
            test_model .test(args,test_loader)
            test_model.eval(args)



if __name__ == '__main__':

    eval()
