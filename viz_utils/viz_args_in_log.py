def print_all_args(args):
    print("--------------------------------------All arguments and their values--------------------------------------")
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    print("----------------------------------------------------------------------------------------------------------")