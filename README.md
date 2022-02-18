# WMTI-watson-py*
Given mean, axial, radial diffusivity, mean, axial, radial kurtosis maps,
calculate WM model parameter maps: 
- f: axonal water Fraction.
- Da: Axonal Diffusivity.
- Depar: Extra-axonal PARallel Diffusivity.
- Deperp: Extra-axonal PERPendicular Diffusivity.
- c2: mean cos ^ 2 of the axon orientation dispersion [1/3, 1]. 
  - c2 = 1 / 3 fully isotropic, 
  - c2 = 1 perfectly parallel. 

c2 is directly related to the Watson distribution concentration parameter kappa(same as in NODDI).

## How To:
WMTI-watson-py requires a path to a folder containing the mean, axial and radial 
diffusivity as well as the mean, axial and radial kurtosis maps, all of which
must be identifiable from the acronyms md, ad, rd, mk, ak, rk, respectively. 
Alternatively, an ordered tuple or list containing the paths to each image can 
be fed, must follow the order: _('md', 'ad', 'rd', 'mk', 'ak', 'rk')_.
    
    import os
    from WMTI_Watson import WMTI_Watson

    input_path = '/home/data/dataset1/subject1/dkifit'
    output_path = '/home/data/dataset1/subject1/wmtiwatson'
    os.mkdir(output_path)
    print(os.listdir(path))
['md.nii.gz', 'ad.nii.gz', 'rd.nii.gz', 'mk.nii.gz', 'ak.nii.gz', 'rk.nii.gz']

    mask = '/home/data/dataset1/subject1/mask.nii.gz'
    wmti = WMTI_Watson(path, mask=mask, invivo=True, nodes=4)
    wmti.fit()
    wmti.save(output_path)
    
    print(os.listdir(output_path))

['Depar.nii.gz', 'Deperp.nii.gz', 'f.nii.gz', 'c2.nii.gz', 'Da.nii.gz']

## _*in testing_