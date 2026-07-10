import argparse
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt

import pydicom

def read_dicom_series(dir_path):
    # Use pydicom to bypass SimpleITK C++ path encoding issues
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            pass
            
    if not slices:
        raise ValueError(f"No valid DICOMs found in {dir_path}")
        
    # Sort by InstanceNumber
    slices.sort(key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
    
    # Extract physical info from the first slice
    ref_ds = slices[0]
    spacing = [float(ref_ds.PixelSpacing[0]), float(ref_ds.PixelSpacing[1]), float(getattr(ref_ds, 'SliceThickness', 1.0))]
    
    # Try to calculate actual Z spacing from ImagePositionPatient
    try:
        z1 = float(slices[0].ImagePositionPatient[2])
        z2 = float(slices[1].ImagePositionPatient[2])
        z_spacing = abs(z1 - z2)
        if z_spacing > 0:
            spacing[2] = z_spacing
    except:
        pass
        
    origin = [0.0, 0.0, 0.0]
    if hasattr(ref_ds, 'ImagePositionPatient'):
        origin = [float(x) for x in ref_ds.ImagePositionPatient]
        
    direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if hasattr(ref_ds, 'ImageOrientationPatient'):
        iop = ref_ds.ImageOrientationPatient
        direction = [float(iop[0]), float(iop[1]), float(iop[2]), float(iop[3]), float(iop[4]), float(iop[5]), 0.0, 0.0, 1.0]

    # Build 3D array
    img_list = []
    for ds in slices:
        image = ds.pixel_array.astype(np.float32)
        intercept = ds.RescaleIntercept if hasattr(ds, 'RescaleIntercept') else 0.0
        slope = ds.RescaleSlope if hasattr(ds, 'RescaleSlope') else 1.0
        hu_image = image * slope + intercept
        img_list.append(hu_image)
        
    volume3d = np.stack(img_list)
    
    # Convert to SimpleITK Image
    sitk_img = sitk.GetImageFromArray(volume3d)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)
    return sitk_img

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    # B-Spline interpolation is best for medical image pixel values, Linear is also fine
    resampler.SetInterpolator(sitk.sitkBSpline) 
    
    return resampler.Execute(image)

def apply_window(image_arr, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image_arr, img_min, img_max)
    # Normalize to 0-255
    windowed = (windowed - img_min) / window_width * 255.0
    return windowed.astype(np.uint8)

def process_case(dir_path, output_dir, name):
    print(f"\n--- Processing {name} ---")
    image = read_dicom_series(dir_path)
    
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    print(f"Original Spacing (X, Y, Z): {[round(s, 2) for s in orig_spacing]}")
    print(f"Original Size: {orig_size}")
    
    # 1. Resample to Isotropic 1.0mm
    new_spacing = [1.0, 1.0, 1.0]
    resampled_img = resample_image(image, new_spacing=new_spacing)
    
    new_spacing_actual = resampled_img.GetSpacing()
    new_size = resampled_img.GetSize()
    print(f"Resampled Spacing (X, Y, Z): {new_spacing_actual}")
    print(f"Resampled Size: {new_size}")
    
    # Calculate how many slices a 10mm lesion takes
    slices_per_10mm_orig = 10.0 / orig_spacing[2]
    slices_per_10mm_new = 10.0 / new_spacing_actual[2]
    print(f"A 10mm lesion occupies ~{slices_per_10mm_orig:.1f} slices originally.")
    print(f"A 10mm lesion occupies exactly {slices_per_10mm_new:.1f} slices after resampling.")
    
    # 2. Get NumPy Array (Z, Y, X)
    img_arr = sitk.GetArrayFromImage(resampled_img)
    
    # 3. Find Lung Bounding Box (simple HU-based thresholding)
    # Lung is typically between -1000 and -400
    z_sum = np.sum((img_arr >= -1000) & (img_arr <= -400), axis=(1, 2))
    total_pixels_per_slice = img_arr.shape[1] * img_arr.shape[2]
    
    # If lung pixels > 5% of a slice, consider it a valid lung slice
    valid_slices = np.where(z_sum > total_pixels_per_slice * 0.05)[0]
    if len(valid_slices) > 0:
        start_z, end_z = valid_slices[0], valid_slices[-1]
    else:
        start_z, end_z = 0, len(img_arr) - 1
        
    print(f"Lung Bounding Box Z-axis: Slice {start_z} to {end_z} (Total: {end_z - start_z + 1} slices)")
    
    # 4. Crop to Lung BB
    cropped_arr = img_arr[start_z:end_z+1]
    
    # 5. Apply Lung Window (W: 1500, L: -600)
    png_arr = apply_window(cropped_arr, -600, 1500)
    
    # Here we would save PNGs, but let's just plot a few to prove it
    os.makedirs(output_dir, exist_ok=True)
    # Save a sample
    plt.imsave(os.path.join(output_dir, f"{name}_sample_slice.png"), png_arr[len(png_arr)//2], cmap='gray')
    
    return {
        "name": name,
        "orig_z_spacing": orig_spacing[2],
        "orig_slices": orig_size[2],
        "new_slices": new_size[2],
        "valid_lung_slices": end_z - start_z + 1,
        "slices_for_10mm": slices_per_10mm_new
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare thick/thin DICOM examples after 1.0 mm resampling.")
    parser.add_argument("--thick-case", required=True, help="Path to an example thick-slice DICOM series.")
    parser.add_argument("--thin-case", required=True, help="Path to an example thin-slice DICOM series.")
    parser.add_argument("--out-dir", required=True, help="Output directory for standardized example PNGs.")
    args = parser.parse_args()

    thick_case = args.thick_case
    thin_case = args.thin_case
    out_dir = args.out_dir
    
    res_thick = process_case(thick_case, out_dir, "Thick_Case_5mm")
    res_thin = process_case(thin_case, out_dir, "Thin_Case_0.6mm")
    
    print("\n--- Summary ---")
    print(f"Thick Case: Orig Spacing {res_thick['orig_z_spacing']:.2f}mm -> 1.0mm. Slices {res_thick['orig_slices']} -> {res_thick['new_slices']}. Valid Lung: {res_thick['valid_lung_slices']}. 10mm = {res_thick['slices_for_10mm']:.1f} slices.")
    print(f"Thin Case: Orig Spacing {res_thin['orig_z_spacing']:.2f}mm -> 1.0mm. Slices {res_thin['orig_slices']} -> {res_thin['new_slices']}. Valid Lung: {res_thin['valid_lung_slices']}. 10mm = {res_thin['slices_for_10mm']:.1f} slices.")
