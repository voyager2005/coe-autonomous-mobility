import torch
from huggingface_hub import hf_hub_download

<<<<<<< Updated upstream
from model.Restorer import Restorer
=======
from model.OneRestore import OneRestore
>>>>>>> Stashed changes
from model.Embedder import Embedder

combine_type = ['clear', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow']

if __name__ == "__main__":
    # push embedder to hub
    embedder = Embedder(combine_type)
    filepath = hf_hub_download(repo_id="gy65896/OneRestore", filename="embedder_model.tar")
    state_dict = torch.load(filepath, map_location="cpu")
    embedder.load_state_dict(state_dict)
    embedder.push_to_hub("gy65896/embedder")

    # push restorer to hub
<<<<<<< Updated upstream
    restorer = Restorer()
=======
    restorer = OneRestore()
>>>>>>> Stashed changes
    filepath = hf_hub_download(repo_id="gy65896/OneRestore", filename="onerestore_cdd-11.tar")
    state_dict = torch.load(filepath, map_location="cpu")
    restorer.load_state_dict(state_dict)
    restorer.push_to_hub("gy65896/restorer")

    # reload
    embedder = Embedder.from_pretrained("gy65896/embedder")
<<<<<<< Updated upstream
    restorer = Restorer.from_pretrained("gy65896/restorer")
=======
    restorer = OneRestore.from_pretrained("gy65896/restorer")
>>>>>>> Stashed changes
