import sys

def get_params():
    model = ''
    try:
        for i, arg in enumerate(sys.argv):
            if arg=='-m':
                model = sys.argv[i+1]
    except Exception:
        print(f'Usage:\n-m: to specify the model\n')
        quit()
    return model
