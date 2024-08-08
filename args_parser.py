import argparse

def parse_args():
    # Initial parsing to get the method
    parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
    parser.add_argument(
        '--method',
        type=str,
        choices=['lpt_x', 
                 'rdm_sampling', 
                 'tgt_sampling', 
                 'rf_classifier', 
                 'mlp_classifier', 
                 'single_fixed_thres'
                 ],
        required=True,
        help="Choose one of the available options: lpt_x, rdm_sampling, tgt_sampling, rf_classifier, mlp_classifier, single_fixed_thres"
    )
    parser.add_argument(
        '--species_set',
        type=str,
        choices=['iucn', 'snt', 'custom'],
        default='iucn',
        help="Choose the species set."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Model path.")
    parser.add_argument("--exp_name", type=str, required=True, default='test', help="Experiment name")
    parser.add_argument("--counter", type=int, default=0, help="Counter value")

    # First parse to get the method
    args, remaining_args = parser.parse_known_args()

    # Create a new parser for the method-specific arguments
    method_parser = argparse.ArgumentParser()

    # Conditionally add arguments based on the method
    if args.method == 'lpt_x':
        method_parser.add_argument(
            '--lpt_level',
            type=int,
            default=0,
            help="Specify the level for lpt-x method"
        )
    elif args.method == 'rdm_sampling':
        # Create a mutually exclusive group for raw_number and factor_presences
        rdm_group = parser.add_mutually_exclusive_group(required=True)
        
        rdm_group.add_argument(
            '--raw_number',
            type=int,
            help="Specify the raw number of absences to generate, e.g. 100, 1000, 10000..."
        )
        rdm_group.add_argument(
            '--factor_presences',
            type=float,
            help="Specify a factor (proportion) of the number of presences to generate absences."
        )

    # Parse all arguments including the conditional ones
    args = parser.parse_args()

    print(args)

    return args

if __name__ == "__main__":
    args = parse_args()