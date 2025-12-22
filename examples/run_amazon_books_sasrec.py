from viberec.run_train import run_experiment_and_upload

if __name__ == '__main__':
    run_experiment_and_upload(
        model_name='SASRec',
        dataset_name='Amazon_Books',
        config_file_list=['examples/config/amazon_books_sasrec.yaml']
    )