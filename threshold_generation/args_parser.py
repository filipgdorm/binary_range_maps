import argparse

def parse_args():
    # Initial parsing to get the method
    parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
    parser.add_argument(
        '--method',
        type=str,
        choices=['lpt_x', 'rdm_sampling', 'tgt_sampling', 'rf_classifier', 'mlp_classifier', 'single_fixed_thres'],
        required=True,
        help="Choose one of the available options: lpt_x, rdm_sampling, tgt_sampling, rf_classifier, mlp_classifier, single_fixed_thres"
    )
    parser.add_argument(
        '--species_set',
        type=str,
        choices=['iucn', 'snt', 'custom'],
        required=True,
        help="Choose "
    )
    parser.add_argument("--model_path", type=str,required=True, help="Model path.")
    parser.add_argument("--result_dir", type=str,required=True, default='test', help="Experiment name")
    parser.add_argument("--counter", type=int, default='test', help="Experiment name")

    # First parse to get the method
    args, remaining_args = parser.parse_known_args()

    # Conditionally add arguments based on the method
    if args.method == 'lpt_x':
        parser.add_argument(
            '--lpt_level',
            type=int,
            required=True,
            help="Specify the level for lpt-x method"
        )
    elif args.method == 'rdm_sampling':
        parser.add_argument(
            '--num_absences',
            type=float,
            required=True,
            help="Specify how many absences to generate for the rdm sampling method."
        )

    # Parse all arguments including the conditional ones
    args = parser.parse_args(remaining_args)

    return args