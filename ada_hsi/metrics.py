import torch 

def psnr(y_true, y_pred, range=1.0):
  return 10*torch.log10( range**2 / torch.mean(torch.square(y_true - y_pred))  )