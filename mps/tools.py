

def take_from_list(O, i):
    if type(O) == list:
        return O[i]
    else:
        return O

DEBUG = True


def log(*args):
    if DEBUG:
        print(*args)