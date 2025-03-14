import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from functools import partial

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from util.img_utils import clear_color
from util.logger import get_logger
from util.img_utils import Blurkernel

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr

def setup_diffusion_environment(model_path='models/ffhq_10m.pt'):
    """
    Set up the environment for diffusion inverse problem.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the pretrained model file
        
    Returns:
    --------
    dict
        Dictionary containing model, device, and logger objects
    """

    try:
        # Install required packages if not already installed
        try:
            import lpips
        except ImportError:
            os.system('pip install lpips')
            import lpips

        # Set up logger
        logger = get_logger()

        # Set up device
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device set to {device_str}.")
        device = torch.device(device_str)

        # Load pretrained model
        model = create_model(
            image_size=256,
            num_channels=128,
            num_res_blocks=1,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="16",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            model_path=model_path
        )
        model = model.to(device)
        model.eval()

        # Return all necessary components
        return {
            'model': model,
            'device': device,
            'logger': logger
        }

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise

