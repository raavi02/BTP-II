# importing the necessary libraries
import os, glob
import micasense.capture as capture
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from skimage.transform import ProjectiveTransform
import numpy as np
from skimage.transform import ProjectiveTransform
import time
import skimage
from skimage.transform import warp,matrix_transform,resize,FundamentalMatrixTransform,estimate_transform,ProjectiveTransform
import micasense.imageutils as imageutils
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from tqdm import tqdm


def get_camera_info(imageNames):
    thecapture = capture.Capture.from_filelist(imageNames)
    # get camera model for future use 
    cam_model = thecapture.camera_model
    # if this is a multicamera system like the RedEdge-MX Dual,
    # we can combine the two serial numbers to help identify 
    # this camera system later. 
    return thecapture, cam_model

def check_and_load_warp_matrices(warp_matrices_filename):
    if Path('./' + warp_matrices_filename).is_file():
        print("Found existing warp matrices for camera", cam_serial)
        load_warp_matrices = np.load(warp_matrices_filename, allow_pickle=True)
        loaded_warp_matrices = []
        for matrix in load_warp_matrices: 
            if panchroCam:
                transform = ProjectiveTransform(matrix=matrix.astype('float64'))
                loaded_warp_matrices.append(transform)
            else:
                loaded_warp_matrices.append(matrix.astype('float32'))

        if panchroCam:
            warp_matrices_SIFT = loaded_warp_matrices
            return warp_matrices_SIFT
        else:
            warp_matrices = loaded_warp_matrices
            return warp_matrices
    else:
        print("No existing warp matrices found. Create them later in the notebook.")
        warp_matrices_SIFT = False
        warp_matrices = False
        return []

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def write_text_on_image(image, text, xywh):
    # Parse xywh coordinates
    x, y, w, h = xywh

    # Load a font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_width, text_height = text_size

    # Calculate the position for the text to be written
    text_x = x + (w - text_width) // 2
    text_y = y + (h + text_height) // 2

    # Write text on image
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base-path', type=str, help='Base path to load the images from')
    parser.add_argument('--image-extension', type=str, metavar='IMG_0124_*.tif', help='Extension of image for which the NDVI value has to be calculated')
    parser.add_argument('--panel-extension', type=str, metavar='IMG_0002_*.tif', help='Extension of panel images')
    parser.add_argument('--use-sharp', action='store_true', help='Use sharp image for masking')
    parser.add_argument('--device', type=str, help='Device to run the code on')  
    parser.add_argument("--output", type=str,
        required=True,
        help=(
            "Directory to store the COCO style masks and NDVI annotated image"
        ),
    )
    parser.add_argument("--model-type", type=str, default = "vit_h",  help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",)
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM checkpoint to use for mask generation.",)
    parser.add_argument("--convert-to-json", action="store_true",help=("Save masks as COCO RLEs in a single json instead of as a folder of PNGs. Requires pycocotools."),)

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.85,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.9,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=1,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=2,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=100,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )



    args = parser.parse_args()
    ##Variation of the micasense code
    imagePath = Path(args.base_path)
    # print(imagePath)
    # these will return lists of image paths as strings 
    imageNames = list(imagePath.glob(args.image_extension))
    imageNames = [x.as_posix() for x in imageNames]

    panelNames = list(imagePath.glob(args.panel_extension))
    panelNames = [x.as_posix() for x in panelNames]

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    ##-------------------------------------------------Camera Specifications------------------------------------------
    thecapture, cam_model = get_camera_info(imageNames)
    if len(thecapture.camera_serials) > 1:
        cam_serial = "_".join(thecapture.camera_serials)
        print(cam_serial)
    else:
        cam_serial = thecapture.camera_serial
    print("Camera model:",cam_model)
    print("Bit depth:", thecapture.bits_per_pixel)
    print("Camera serial number:", cam_serial)
    print("Capture ID:",thecapture.uuid)

    # determine if this sensor has a panchromatic band 
    if cam_model == 'RedEdge-P' or cam_model == 'Altum-PT':
        panchroCam = True
    else:
        panchroCam = False
        panSharpen = False
    
    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.49]*len(thecapture.eo_band_names()) #RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)  
        irradiance_list = panelCap.panel_irradiance(panel_reflectance_by_band) + [0] # add to account for uncalibrated LWIR band, if applicable
        img_type = "reflectance"
        # thecapture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if thecapture.dls_present():
            img_type='reflectance'
            irradiance_list = thecapture.dls_irradiance() + [0]
            # thecapture.plot_undistorted_reflectance(thecapture.dls_irradiance())
        else:
            img_type = "radiance"
            # thecapture.plot_undistorted_radiance() 
            irradiance_list = None

    if panchroCam:
        warp_matrices_filename = cam_serial + "_warp_matrices_SIFT.npy"
        warp_matrices_SIFT = check_and_load_warp_matrices(warp_matrices_filename)
    else:
        warp_matrices_filename = cam_serial + "_warp_matrices_opencv.npy"
        warp_matrices = check_and_load_warp_matrices(warp_matrices_filename)
    
    #----------------------------------------Alignment-----------------------------------------------------
    if panchroCam: 
        # set to True if you'd like to ignore existing warp matrices and create new ones
        regenerate = False
        st = time.time()
        if not warp_matrices_SIFT or regenerate:
            print("Generating new warp matrices...")
            warp_matrices_SIFT = thecapture.SIFT_align_capture(min_matches = 10)
            
        sharpened_stack, upsampled = thecapture.radiometric_pan_sharpened_aligned_capture(warp_matrices=warp_matrices_SIFT, irradiance_list=irradiance_list, img_type=img_type)
        
    # we can also use the Rig Relatives from the image metadata to do a quick, rudimentary alignment 
    #     warp_matrices0=thecapture.get_warp_matrices(ref_index=5)
    #     sharpened_stack,upsampled = radiometric_pan_sharpen(thecapture,warp_matrices=warp_matrices0)

    print("Pansharpened shape:", sharpened_stack.shape)
    print("Upsampled shape:", upsampled.shape)
    # re-assign to im_aligned to match rest of code 
    im_aligned = upsampled
    et = time.time()
    elapsed_time = et - st
    print('Alignment and pan-sharpening time:', int(elapsed_time), 'seconds')

    ##Save warp matrices
    if panchroCam:
        working_wm = warp_matrices_SIFT
    else:
        working_wm = warp_matrices
    if not Path('./' + warp_matrices_filename).is_file() or regenerate:
        temp_matrices = []
        for x in working_wm:
            if isinstance(x, np.ndarray):
                temp_matrices.append(x)
            if isinstance(x, skimage.transform._geometric.ProjectiveTransform):
                temp_matrices.append(x.params)
        np.save(warp_matrices_filename, np.array(temp_matrices, dtype=object), allow_pickle=True)
        print("Saved to", Path('./' + warp_matrices_filename).resolve())
    else:
        print("Matrices already exist at",Path('./' + warp_matrices_filename).resolve())

    nir_band = thecapture.band_names_lower().index('nir')
    red_band = thecapture.band_names_lower().index('red')
    ndvi = (im_aligned[:,:,nir_band] - im_aligned[:,:,red_band]) / (im_aligned[:,:,nir_band] + im_aligned[:,:,red_band])

    rgb_band_indices = [thecapture.band_names_lower().index('red'),
                thecapture.band_names_lower().index('green'),
                thecapture.band_names_lower().index('blue')]
    
    im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],im_aligned.shape[2]), dtype=np.float32)
    im_min = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
    im_max = np.percentile(im_aligned[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values

    if panchroCam:
        im_display_sharp = np.zeros((sharpened_stack.shape[0],sharpened_stack.shape[1],sharpened_stack.shape[2]), dtype=np.float32 )
        im_min_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 0.5)  # modify these percentiles to adjust contrast
        im_max_sharp = np.percentile(sharpened_stack[:,:,rgb_band_indices].flatten(), 99.5)  # for many images, 0.5 and 99.5 are good values
    
    for i in rgb_band_indices:
        im_display[:,:,i] =  imageutils.normalize(im_aligned[:,:,i], im_min, im_max)
        if panchroCam: 
            im_display_sharp[:,:,i] = imageutils.normalize(sharpened_stack[:,:,i], im_min_sharp, im_max_sharp)

    rgb = im_display[:,:,rgb_band_indices]

    if panchroCam:
        rgb_sharp = im_display_sharp[:,:,rgb_band_indices]
 
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    print(f"Model loaded to device: {args.device}")
    output_mode = "coco_rle" if args.convert_to_json else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    # generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)
    image_name = args.image_extension[:8]
    save_base = os.path.join(args.output, image_name)
    if args.use_sharp:
        masks = generator.generate(rgb_sharp)
        rgb_image = rgb_sharp.copy()
    else:
        masks = generator.generate(rgb)
        rgb_image = rgb.copy()
    
    print("Segmentation done!")
    #Saving the masks
    image_name = args.image_extension[:8]
    save_base = os.path.join(args.output, image_name) 
    if output_mode == "binary_mask":
        os.makedirs(save_base, exist_ok=False)
        write_masks_to_folder(masks, save_base)
    else:
        save_file = save_base + ".json"
        masks_json = masks.copy()
        for mask in masks_json:
            mask["segmentation"] = mask["segmentation"].tolist()
        masks_json = json.dumps(masks_json)
        with open(save_file, "w") as f:
            json.dump(masks_json, f)
        
    
    NDVI = []
    #calculating NDVI
    print("Processing masks!")
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    for i in tqdm(range(len(masks))):
        NDVI.append(np.average(ndvi[masks[i]["segmentation"]]))

    
    for mask, vi in zip(masks, NDVI):
        xywh = mask["bbox"]
        color_mask = np.concatenate([np.random.random(2), [0.35]])
        rgb_image[mask["segmentation"]] = color_mask
        text = str(round(vi,2))
        rgb_image = write_text_on_image(rgb_image, text, xywh)

    

    plt.imsave(save_base + '-NDVI.png', rgb_image) 
    print("Annotated NDVI file saved to: " + save_base + '-NDVI.png')
    

    

    

        
        

        

        

    