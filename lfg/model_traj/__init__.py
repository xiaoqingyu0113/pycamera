import os
import glob
import importlib
import torch

# Get the directory of the current file
current_dir = os.path.dirname(__file__)
py_files = glob.glob(os.path.join(current_dir, "*.py"))
module_names = [os.path.splitext(os.path.basename(f))[0] for f in py_files if not f.endswith('__init__.py')]

# Import all modules
imported_modules =  [importlib.import_module(f".{module_name}", package=__name__) for module_name in module_names]


# Import all functions and classes from the imported modules
all_tmp = []
for m in imported_modules:
    for attr_name in dir(m):
        attr = getattr(m, attr_name)

        if isinstance(attr, type) and issubclass(attr, torch.nn.Module):
            # print(f'...{attr} imported as Module')
            globals()[attr_name] = getattr(m, attr_name)
            all_tmp.append(attr_name)
        elif 'autoregr' in attr_name:
            # print(f'...{attr} imported as AutoRegressive')
            globals()[attr_name] = attr
            all_tmp.append(attr_name)
# update
__all__ = all_tmp

