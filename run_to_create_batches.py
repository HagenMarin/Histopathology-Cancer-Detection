from data_loading import create_batches

def main():
    batch_size = 128
    create_batches(batch_size)

if __name__=='__main__':
    main()