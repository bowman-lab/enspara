import argparse
import os


class readable_dir(argparse.Action):
    """Argparse action that determines if the option given points to a
    directory that exists and its writable.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = os.path.dirname(os.path.abspath(values))
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, values)
        else:
            raise argparse.ArgumentTypeError(
                "readable_dir:{0} is not a readable dir".format(
                    prospective_dir))
