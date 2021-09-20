import sys
import argparse


def identify_app(argv):

    parser = argparse.ArgumentParser(
        prog='enspara',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main entry point for enspara apps.")

    parser.add_argument(
        "appname",
        choices={'cluster', 'implied', 'reassign'},
        help="Name of the application.")

    parser.add_argument(
        "appargs", nargs="*",
        help="Subsequent arguments to the app (add subcommand for more).")

    helpstack = []
    for h in ['--help', '-h']:
        while h in argv and argv.index(h) != 1:
            argv.remove(h)
            helpstack.append(h)

    args = parser.parse_args(argv[1:])

    if args.appname == 'cluster':
        from enspara.apps.cluster import main
    elif args.appname == 'implied':
        from enspara.apps.implied_timescales import main
    elif args.appname == 'reassign':
        from enspara.apps.reassign import main

    args.main = main
    args.appargs.extend(helpstack)

    return args


def main(argv=None):

    args = identify_app(argv)

    try:
        args.main(args.appargs)
    except Exception as e:
        message = ("An unexpected error has occurred; please consider filing "
                   "an issue at our issue tracker:\n"
                   "https://github.com/bowman-lab/enspara/issues")
        print(message, file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
