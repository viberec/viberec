import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viberec.data import preprocess_and_upload

def main():
    parser = argparse.ArgumentParser(description='Preprocess Data and Upload to Hugging Face')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--repo_id', '-r', type=str, required=True, help='HuggingFace Repo ID to push to')
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='Model name (default: SASRec)')
    parser.add_argument('--config_files', type=str, default=None, help='Config files (space separated)')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace Token')

    args, unknown = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    # Parse unknown args to config_dict
    config_dict = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith('--'):
            key = key[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                value = unknown[i+1]
                if value.lower() == 'true': value = True
                elif value.lower() == 'false': value = False
                elif value.lower() == 'none': value = None
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                config_dict[key] = value
                i += 2
            else:
                config_dict[key] = True
                i += 1
        else:
            i += 1

    preprocess_and_upload(
        dataset_name=args.dataset,
        repo_id=args.repo_id,
        model_name=args.model,
        config_file_list=config_file_list,
        config_dict=config_dict,
        hf_token=args.hf_token
    )

if __name__ == '__main__':
    main()
