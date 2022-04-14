function [datapath]=WMTI_Watson_wrapper(datapath, maskfilepath, invivo_flag)

    % loads nifti files for brain mask, DTI and DKI metrics, runs
    % WMTI-Watson calculation, writes nifti files for model parameters (f,
    % Da, Depar, Deperp, c2) and fit exitflag for QA purposes (exitflags).
    
    % inputs: 
    % datapath: full path to where the DTI and DKI maps (mean/axial/radial diffusivity and kurtosis) are stored.
    % maskfilepath: full path to where a brain mask is stored. if not
    % supplied, a mask will be created based on non-NaN values in the md
    % map.
    % invivo_flag: boolean. true (invivo), false (exvivo). Default: in
    % vivo.
    
    % set invivo/exvivo flag
    if isempty(invivo_flag)
        invivo_flag = true; % by default: in vivo
    end
    
    % load DTI and DKI maps - you can update this to your own filename
    % nomenclature
    nii = load_untouch_nii(fullfile(datapath,'md.nii')); md = double(nii.img);
    nii = load_untouch_nii(fullfile(datapath,'ad.nii')); ad = double(nii.img);
    nii = load_untouch_nii(fullfile(datapath,'rd.nii')); rd = double(nii.img);
    nii = load_untouch_nii(fullfile(datapath,'mk.nii')); mk = double(nii.img);
    nii = load_untouch_nii(fullfile(datapath,'ak.nii')); ak = double(nii.img);
    nii = load_untouch_nii(fullfile(datapath,'rk.nii')); rk = double(nii.img);
    
    % load ROI mask
    if isempty(maskfilepath)
        mask = ~isnan(md);
    else
        nii_mask = load_untouch_nii(maskfilepath);
        mask = nii_mask.img; mask = mask>0;
    end

    % call main function to calculate parametric maps for WM model
    [f,Da,Depar,Deperp,c2,exitflags] = WMTI_Watson_maps(md,ad,rd,mk,ak,rk,mask,invivo_flag);%
    
    % write output nifti files for WM model parameters
    nii_temp = nii;
    nii_temp.hdr.dime.datatype = 16;
    nii_temp.hdr.dime.scl_slope = 0;
    nii_temp.img = f;
    save_untouch_nii(nii_temp,fullfile(datapath,'f.nii'))
    nii_temp.img = Da;
    save_untouch_nii(nii_temp,fullfile(datapath,'Da.nii'))
    nii_temp.img = Depar;
    save_untouch_nii(nii_temp,fullfile(datapath,'Depar.nii'))
    nii_temp.img = Deperp;
    save_untouch_nii(nii_temp,fullfile(datapath,'Deperp.nii'))
    nii_temp.img = c2;
    save_untouch_nii(nii_temp,fullfile(datapath,'c2.nii'))
    nii_temp.img = exitflags;
    save_untouch_nii(nii_temp,fullfile(datapath,'exitflag.nii'))
end