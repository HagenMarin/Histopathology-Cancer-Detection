import sys

def get_params():
    model = ''
    save_name = 'default'
    try:
        for i, arg in enumerate(sys.argv):
            match arg:
                case '-m':
                    model = sys.argv[i+1]
                case '-n':
                    save_name = sys.argv[i+1]
                case '-help':
                    print_help_msg()
                    quit()
                case 'help':
                    print_help_msg()
                    quit()
    except Exception:
        print_help_msg()
        quit()
    return model, save_name

def print_help_msg():
    print(f'Usage:\n-m: to specify the model\n-n: to specify the save name of the trained model\n')
