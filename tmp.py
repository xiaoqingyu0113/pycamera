from colorama import Fore, Style, init
from collections import defaultdict

init(autoreset=True)
def default_styles():
    return Fore.WHITE

styles = defaultdict(default_styles)
styles.update({'name': Fore.RED,
                'lr_init': Fore.GREEN,
                'continue_training': Fore.YELLOW,
                })

d = {'name': 'pyhtune', 'lr_init': 0.01, 'continue_training': False,'random': 'random',}


for kk, vv in d.items():
    print(styles[kk] + f'{kk}: {vv}'+ Style.RESET_ALL)