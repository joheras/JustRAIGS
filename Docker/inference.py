import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from fastai.vision.all import *
import fastai
import os
# from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
# from numpy import asarray


DEFAULT_GLAUCOMATOUS_FEATURES_ACRONYM = {
    "appearance neuroretinal rim superiorly": "ANRS",
    "appearance neuroretinal rim inferiorly": "ANRI",
    "retinal nerve fiber layer defect superiorly": "RNFLDS",
    "retinal nerve fiber layer defect inferiorly": "RNFLDI",
    "baring of the circumlinear vessel superiorly": "BCLVS",
    "baring of the circumlinear vessel inferiorly": "BCLVI",
    "nasalization of the vessel trunk": "NVT",
    "disc hemorrhages": "DH",
    "laminar dots": "LD",
    "large cup": "LC",
}

def ratioDiskCup(image,processor,model):
    image_new = asarray(Image.open(image))
    (h,w)=image_new.shape[:2]

    inputs = processor(image_new, return_tensors="pt")
    inputs.to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image_new.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    pred_disc_cup = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    
    
    disk = np.sum(pred_disc_cup == 1)
    cup = np.sum(pred_disc_cup == 2)
    ratio = cup/disk
    return torch.Tensor([1-ratio,ratio])

def run():
    
    # Segmenting cup and disk
    # processor = AutoImageProcessor.from_pretrained("/opt/app/segformer_for_optic_disc_cup_segmentation",device="cuda", local_files_only=True)
    # model = SegformerForSemanticSegmentation.from_pretrained("/opt/app/segformer_for_optic_disc_cup_segmentation", local_files_only=True)
    # model.cuda()
    _show_torch_cuda_info()
    
    # learn2 = load_learner('efficientnetv2_rw_s.pkl')
    learn2 = load_learner('swin_base_patch4_window12_384.ms_in1k.pkl')
    learn1 = load_learner('convnext_base_384_in22ft1k.pkl')
    learn_multi_label = load_learner('convnext_base_384_in22ft1k_multi_labelv3.pkl')
    learn_multi_label.remove_cbs(learn_multi_label.cbs)
    learn_multi_label.remove_cbs([Recorder,ProgressCallback,SaveModelCallback])
    learn1.model.cuda()
    learn1.dls.cuda()
    learn2.model.cuda()
    learn2.dls.cuda()
    learn_multi_label.model.cuda()
    learn_multi_label.dls.cuda()    

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant


        print(f"Running inference on {jpg_image_file_name}")

        # Normal version
        # is_referable_glaucoma_likelihood = float(((learn1.predict(jpg_image_file_name)[2]+learn2.predict(jpg_image_file_name)[2])/2)[1])
        # TTA version
        is_referable_glaucoma_likelihood = float(((learn1.tta(dl=learn1.dls.test_dl([jpg_image_file_name]))[0][0]+learn2.tta(dl=learn2.dls.test_dl([jpg_image_file_name]))[0][0])/2)[1])
        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.45
        # if(len(multi_label)>0):
        #     is_referable_glaucoma_likelihood = 1-float(prediction[2][7])
        #     is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.45
        # else:
        #     is_referable_glaucoma=False
        #     is_referable_glaucoma_likelihood = float(prediction[2][7])
        
        if is_referable_glaucoma:
            # Old version
            prediction = learn_multi_label.predict(jpg_image_file_name)
            multi_label = [clase for (pred,clase) in zip(prediction[2],learn_multi_label.dls.vocab) if pred>0.5]
            # TTA version
            # res = learn_multi_label.tta(dl=learn_multi_label.dls.test_dl([jpg_image_file_name]))
            # multi_label = [clase for (pred,clase) in zip(res[0][0],learn_multi_label.dls.vocab) if pred>0.5]
            features = {
                k: (DEFAULT_GLAUCOMATOUS_FEATURES_ACRONYM[k] in multi_label)
                for k in DEFAULT_GLAUCOMATOUS_FEATURES.keys()
            }
        else:
            features = None

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
