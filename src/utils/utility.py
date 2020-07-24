import os
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if count == 1:
        print("")
    print("\r[{}] {}% {} of {} {} ".format(bar, percents, count, total, status), end='')
    if count == total:
        print("")
    os.sys.stdout.flush()