#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
import fastai
import timm
import albumentations as A

# In[2]:


import torch
torch.cuda.set_device(2)


# In[2]:

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())
model_name=args["model"]#'convnext_base_384_in22ft1k'


df = pd.read_csv('prepared_binary_dataset.csv')
df["pathMask"]=df["path"].apply(lambda x: x.replace("train/", "trainMask/"))

# In[3]:


def is_valid(path):
        name = path[0]
        return (dfNew[dfNew['Eye ID']==name])['label'].values[0]=='valid'

db = DataBlock(blocks = (ImageBlock, CategoryBlock),
             splitter=FuncSplitter(is_valid),
             get_x = ColReader(-1),
             get_y=ColReader(1),
             item_tfms = [Resize(384)], # CropPad(200,200)
             batch_tfms=[*aug_transforms(size=384, min_scale=0.75,do_flip=True,flip_vert=True,
                  max_rotate=2.,max_zoom=1.1, max_warp=0.05,p_affine=0.9, p_lighting=0.8), 
                         Normalize.from_stats(*imagenet_stats)])


# In[ ]:

for i in range(0,10):
    dfNew = pd.concat([df[(df.label=='training') & (df['Final Label']=='NRG')].sample(3000),
                   df[(df.label=='training') & (df['Final Label']=='RG')],
                   df[(df.label=='valid') & (df['Final Label']=='NRG')].sample(818),
                   df[(df.label=='valid') & (df['Final Label']=='RG')]])
    




    dls = db.dataloaders(dfNew.values,bs=8)




    # In[ ]:


    from fastai.vision.all import *
    callbacks = [
        ShowGraphCallback(),
        #EarlyStoppingCallback(patience=5),
        SaveModelCallback(fname=model_name+"onlymask",monitor='cohen_kappa_score'),
        ReduceLROnPlateau(patience=2), 
        #ProgressiveResizingRandAugment([0,100,200,300,350],[224,384,512,640,640],[128,224,384,512,512],[0,2,4,6,8],128,'resnetrsUpretina')
    ]


    # In[ ]:

    if(i==0):
        learn = Learner(dls,timm.create_model(model_name,num_classes=2,pretrained=True),
                    metrics=[accuracy,Precision(),Recall(),F1Score(),CohenKappa(weights='quadratic')],cbs=callbacks,
                    loss_func= FocalLossFlat()).to_fp16()
    else:
        learn.dls=dls


    learn.fine_tune(3,base_lr=3e-4)


dfNew = pd.concat([df[(df.label=='training') & (df['Final Label']=='NRG')].sample(3000),
                   df[(df.label=='training') & (df['Final Label']=='RG')],
                   df[(df.label=='valid')]])

dls = db.dataloaders(dfNew.values,bs=16)
learn.save(model_name+"onlymask")
learn2 = Learner(dls,timm.create_model(model_name,num_classes=2,pretrained=True)).to_fp16()
learn2.load(model_name+"onlymask")
learn2.export(model_name+'_onlymask.pkl')
learn2.dls=dls
preds,gt=learn2.get_preds()
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

with open('results.txt',mode="a") as f:
    f.write(model_name+' only mask :\n')
    f.write("Accuracy: "+str(accuracy_score(gt,np.argmax(preds,axis=1)))+"\n")
    f.write("Precision: "+str(precision_score(gt,np.argmax(preds,axis=1)))+"\n")
    f.write("Recall: "+str(recall_score(gt,np.argmax(preds,axis=1)))+"\n")
    f.write("F1-score: "+str(f1_score(gt,np.argmax(preds,axis=1)))+"\n")
    f.write('\n')


# In[ ]:


