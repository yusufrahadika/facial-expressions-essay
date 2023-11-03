import os

import gdown

output_path = 'output'
is_output_dir_exists = os.path.exists(output_path) and not os.path.isfile(output_path)
if not is_output_dir_exists:
    os.mkdir(output_path)


output_file = 'output/best-result.pt'
if not os.path.exists(output_file):
    url = 'https://drive.google.com/uc?id=1O1ehV1z3RkBGatqo53hhcSWM_GNMzke4'
    gdown.download(url, output_file, quiet=False)
