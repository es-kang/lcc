import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--gpu", type=str, default=4)
    parser.add_argument("--rp", type=int, default=1, help='The number of repetition')
    parser.add_argument("--fold", type=int, default=1)

    # Dataset
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--bs", type=int, default=32, help='Batch size')
    parser.add_argument("--input_size", type=int, default=200, help="The number of ROIs")

    # parser.add_argument("--df", type=str, default='gumbel', choices=['gumbel', 'sigmoid'])
    parser.add_argument("--df_tau", type=float, default=5.0)
    parser.add_argument("--df_soft", type=str2bool, default='True')

    # parser.add_argument("--intra", type=int, default=6)
    parser.add_argument("--num_stack_sa", type=int, default=2)
    parser.add_argument("--num_heads_sa", type=int, default=10)
    parser.add_argument("--d_ff", type=int, default=128)

    parser.add_argument("--num_proto_td", type=int, default=1)
    parser.add_argument("--num_proto_asd", type=int, default=1)

    parser.add_argument("--lr_df", type=float, default=0.0005)
    parser.add_argument("--lr_tf", type=float, default=0.0005)
    parser.add_argument("--lr_pr", type=float, default=0.0005)
    parser.add_argument("--lr_dc", type=float, default=0.0005)

    parser.add_argument("--lr_dc2", type=float, default=0.0001)

    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--h_cls", type=float, default=1.0)
    parser.add_argument("--h_rec", type=float, default=1.0)
    parser.add_argument("--h_cls2", type=float, default=0.1)
    parser.add_argument("--h_rec2", type=float, default=1.0)
    parser.add_argument("--h_sca", type=float, default=0.5)
    parser.add_argument("--h_roi", type=float, default=1.0)

    ARGS = parser.parse_args()

    return ARGS