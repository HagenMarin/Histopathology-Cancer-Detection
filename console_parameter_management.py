import sys

def get_params():
    model = ''
    save_name = 'default'
    pretrained = False
    pretrained_path = ''
    early_stopping = True
    try:
        for i, arg in enumerate(sys.argv):
            match arg:
                case '-m':
                    model = sys.argv[i+1]
                case '-n':
                    save_name = sys.argv[i+1]
                case '-p':
                    pretrained = True
                    pretrained_path = sys.argv[i+1]
                case '-nes':
                    early_stopping = False
                case '-help':
                    print_help_msg()
                    quit()
                case 'help':
                    print_help_msg()
                    quit()
    except Exception:
        print_help_msg()
        quit()
    return model, save_name, pretrained, pretrained_path, early_stopping

def print_help_msg():
    print(f'Usage:\n-m: to specify the model\n-n: to specify the save name of the trained model\n-p: to use a pretrained model\n-nes: to not use early stopping')
