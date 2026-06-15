# fix_ckpt.py
import torch

classes = ['A','B','CH','D','E','ERII','F','G','H','Htemdeg',
           'I','J','K','L','M','N','O','OU','P','R','S','SH',
           'SHCH','T','TS','U','V','Y','YA','YE','YO','YU',
           'Z','Ztemdeg','hI']

ckpt = torch.load('checkpoints/best_stgcn.pt', map_location='cpu')
ckpt['classes'] = classes
torch.save(ckpt, 'checkpoints/best_stgcn.pt')
print("Done. Classes saved to checkpoint.")