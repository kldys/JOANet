
import warnings
import numpy as np
import torch
import torch.nn as nn
from model import SISR
from skimage import io, color
import os

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_sisr():
    with torch.no_grad():
        torch.cuda.empty_cache()
        net = SISR()
        net = nn.DataParallel(net)
        torch.cuda.empty_cache()
        net.load_state_dict(torch.load('./model/BTM_SISR4.pth'))

        psnr_ac = 0
        data_num = 1
        for j in range(data_num):
            test = io.imread('./test_imgs/BTM/' + '%d' % (j+1) + '.png')
            #label = io.imread('./test_imgs/mri_hr/' + '%d' % (j+1) + '.png')
            test = torch.FloatTensor(test/255).unsqueeze(0).unsqueeze(0).cuda()
            #label = torch.FloatTensor(label/255).unsqueeze(0).unsqueeze(0).cuda()

            output = net(test)
            output = torch.clamp(output, 0.0, 1.0)
            #loss = (output * 255 - label * 255).pow(2).sum() / (output.shape[2] * output.shape[3])
            #psnr = 10 * np.log10(255 * 255 / loss.item())
            #print(psnr)
            #psnr_ac += psnr


            output = output.squeeze(0).squeeze(0).cpu()

            output_array = np.array(output * 255).astype(np.float32)

            SR_image = output_array
            SR_image = np.clip(SR_image, a_min=0., a_max=255.)
            SR_image = SR_image.astype(np.uint8)
            io.imsave('./test_imgs/BTM/%d.png' % (j+1),SR_image)
        #print(psnr_ac/data_num)
run_sisr()