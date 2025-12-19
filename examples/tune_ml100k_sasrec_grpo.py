import argparse
from viberec.tuning import run_bayes_tuning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--output_file', type=str, default='optuna_results.csv', help='output csv file')
    args, _ = parser.parse_known_args()

    run_bayes_tuning(
        n_trials=args.n_trials,
        output_file=args.output_file
    )

if __name__ == '__main__':
    main()
