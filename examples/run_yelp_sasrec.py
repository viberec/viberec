from viberec.run_train import run_experiment_and_upload

if __name__ == '__main__':
    run_experiment_and_upload(
        model_name='SASRec',
        dataset_name='yelp',
        config_file_list=['examples/config/yelp_sasrec.yaml']
    )
