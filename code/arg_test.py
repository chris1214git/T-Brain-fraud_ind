import argparse 

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--th',type=float,default=0.37,help='threshold of output, to get a better F1 score')
    parser.add_argument('-it','--iteration',type=int,default=2000,help='iteration of catboost')
    parser.add_argument('-l2','--l2_leaf_reg',type=int,default=3,help='l2_leaf_reg of catboost')
    parser.add_argument('-d','--data_list',type=str,default='data_list_FE_AN',help='data_list')
    parser.add_argument('-e','--delete_list',type=str,default='delete_list_overfit1',help='delete_list')

    return parser.parse_args()

if __name__ == '__main__':
    args = process_command()
    print(args.text)