import random
import string


def generate_random_filename(extension, length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)) + extension