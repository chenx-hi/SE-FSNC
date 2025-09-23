def parse_arguments():
    parser = argparse.ArgumentParser()

    # GIN parameters
    parser.add_argument('--dataset_name', type=str, default="ENZYMES",
                        help='name of dataset')
    parser.add_argument('--tree_height', type=int, default=2)

    parser.add_argument('--baseline_mode', type=str, default=None,
                        help='baseline')

    parser.add_argument('--N_way', type=int, default=2)
    parser.add_argument('--K_shot', type=int, default=10)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch', type=float, default=128)

    parser.add_argument('--gin_layer', type=int, default=2)
    parser.add_argument('--gin_hid', type=int, default=256)
    parser.add_argument('--aug1', type=str, default='identity')
    parser.add_argument('--aug2', type=str, default='feature_mask')
    parser.add_argument('--t', type=float, default=0.2)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--epoch_num', type=int, default=1000)

    parser.add_argument('--use_select_sim', type=bool, default=False)
    parser.add_argument('--gen_train_num', type=int, default=0) #500
    parser.add_argument('--gen_test_num', type=int, default=0)  #20

    parser.add_argument('--save_test_emb', type=bool, default=True)
    parser.add_argument('--test_mixup', type=bool, default=True)
    parser.add_argument('--num_token', type=int, default=1)

    args = parser.parse_args()
    return args

def parse_arguments():
    parser = argparse.ArgumentParser()

    # GIN parameters
    parser.add_argument('--dataset_name', type=str, default="Letter_high",
                        help='name of dataset')
    parser.add_argument('--tree_height', type=int, default=2)

    parser.add_argument('--baseline_mode', type=str, default=None,
                        help='baseline')

    parser.add_argument('--N_way', type=int, default=4)
    parser.add_argument('--K_shot', type=int, default=10)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch', type=float, default=128)

    parser.add_argument('--gin_layer', type=int, default=2)
    parser.add_argument('--gin_hid', type=int, default=128)
    parser.add_argument('--aug1', type=str, default='identity')
    parser.add_argument('--aug2', type=str, default='feature_mask')
    parser.add_argument('--t', type=float, default=0.05)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--epoch_num', type=int, default=1000)

    parser.add_argument('--use_select_sim', type=bool, default=False)
    parser.add_argument('--gen_train_num', type=int, default=0) #500
    parser.add_argument('--gen_test_num', type=int, default=0)  #20

    parser.add_argument('--save_test_emb', type=bool, default=True)
    parser.add_argument('--test_mixup', type=bool, default=True)
    parser.add_argument('--num_token', type=int, default=1)

    args = parser.parse_args()
    return args
