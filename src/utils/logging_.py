#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.

import os
import pdb
import sys
import torch.distributed as dist


def main_print(content, logger):
    if dist.get_rank() <= 0:
        print(content)
        logger.info(content)


# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
