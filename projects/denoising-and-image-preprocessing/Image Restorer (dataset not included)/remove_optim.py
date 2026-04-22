import torch, argparse
<<<<<<< Updated upstream
from model.Restorer import Restorer
=======
from model.OneRestore import OneRestore
>>>>>>> Stashed changes
from model.Embedder import Embedder

parser = argparse.ArgumentParser()

<<<<<<< Updated upstream
parser.add_argument("--type", type=str, default = 'Restorer')
    parser.add_argument("--input-file", type=str, default = './ckpts/restorer_model.tar')
    parser.add_argument("--output-file", type=str, default = './ckpts/restorer_model.tar')

args = parser.parse_args()

if args.type == 'Restorer':
    restorer = Restorer().to("cuda" if torch.cuda.is_available() else "cpu")
=======
parser.add_argument("--type", type=str, default = 'OneRestore')
parser.add_argument("--input-file", type=str, default = './ckpts/onerestore_cdd-11.tar')
parser.add_argument("--output-file", type=str, default = './ckpts/onerestore_cdd-11.tar')

args = parser.parse_args()

if args.type == 'OneRestore':
    restorer = OneRestore().to("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> Stashed changes
    restorer_info = torch.load(args.input_file, map_location='cuda:0')
    weights_dict = {}
    for k, v in restorer_info['state_dict'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    restorer.load_state_dict(weights_dict)
    torch.save(restorer.state_dict(), args.output_file)
elif args.type == 'Embedder':
    combine_type = [
        'clear', 'low', 'haze', 'blur', 'noise', 
        'low_haze', 'low_blur', 'low_noise', 'haze_blur', 
        'haze_noise', 'low_haze_blur', 'low_haze_noise', 'low_haze_blur_noise'
    ]
    embedder = Embedder(combine_type).to("cuda" if torch.cuda.is_available() else "cpu")
    embedder_info = torch.load(args.input_file)
    embedder.load_state_dict(embedder_info['state_dict'])
    torch.save(embedder.state_dict(), args.output_file)
else:
    print('ERROR!')

