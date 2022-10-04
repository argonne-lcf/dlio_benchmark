"""
   Copyright 2021 UChicago Argonne, LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os

from src.utils.argument_parser import ArgumentParser


def progress(count, total, status=''):
    _arg_parser = ArgumentParser.get_instance()
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if _arg_parser.args.debug:
        if count == 1:
            print("")
        print("\r[{}] {}% {} of {} {} ".format(bar, percents, count, total, status), end='')
        if count == total:
            print("")
        os.sys.stdout.flush()
