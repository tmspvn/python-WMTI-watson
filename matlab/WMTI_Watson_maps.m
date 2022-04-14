function [f,Da,Depar,Deperp,c2,exitflags] = WMTI_Watson_maps(md,ad,rd,mk,ak,rk,mask,invivo_flag)

% given md, ad, rd, mk, ak, rk (mean,axial,radial diffusivity, mean,axial,radial kurtosis) maps, 
% calculate WM model parameter maps: 
% f (axonal water fraction), Da (axonal diffusivity), Depar, Deperp (extra-axonal
% parallel and perpendicular diffusivities), c2 (mean cos^2 of the axon
% orientation dispersion: c2=1/3 fully isotropic, c2=1 perfectly parallel)
% c2 is directly related to the Watson distribution concentration parameter
% kappa (same as in NODDI)

% All diffusivities in um2/ms
% md, ad, rd should also be in um2/ms, otherwise converted here

% mask: brain or ROI mask
% invivo_flag: boolean, flag for in vivo (true) or ex vivo (false)

% I. Jelescu, July 2021

lb = [0 0 0 0 0]; % fit lower bound for model parameters [f, Da, Depar, Deperp, kappa]
if invivo_flag
    ub = [1 3 3 3 128]; % fit upper bound for model parameters, in vivo
    x0 = [0.9 2.2 1.6 0.7 7]; % initial guess, in vivo
    md_ub = 2.5;
else
    ub = [1 2 2 2 128]; % fit upper bound for model parameters, ex vivo
    x0 = [0.9 1.6 1 0.4 7]; % initial guess, ex vivo
    md_ub = 1.8; % upper bound on md to avoid CSF contamination
end

% Check md in um2/ms and not mm2/s, if not, convert
if max(md(:))<1e-2
    md = md*1e3; ad = ad*1e3; rd = rd*1e3;
end

% filter out voxels with unrealistic tensor values
filter = (md<md_ub) & (rk>0) & (rk<10) & (mk>0) & (mk<10); 
roi = mask & filter; % exclude voxels with unphysical values from calculation

% calculate signal moments

Wpar = ak.*(ad./md).^2;
Wperp = rk.*(rd./md).^2;

D0 = md; % mean diffusivity
D2 = 2/3*(ad-rd);
W0 = mk;
W2 = 1/7*(3*Wpar + 5*mk - 8*Wperp);
W4 = 4/7*(Wpar - 3*mk + 2*Wperp);

% initialize model parameter maps
f = NaN*ones(size(md));
Da = f; Depar = f; Deperp = f; kappa = f; exitflags = f;

options = optimoptions('lsqnonlin','MaxIterations',5000,'Display','off');

cnt = sum(~isnan(f(:)));

for k=1:size(md,3)
    
    disp(cnt/sum(roi(:)))
    
    for j = 1:size(md,2)
        
        parfor i=1:size(md,1)
            
            if roi(i,j,k)>0 
                
                cnt = cnt+1;
                
                moments = [D0(i,j,k) D2(i,j,k) W0(i,j,k) W2(i,j,k) W4(i,j,k)];
                
                [x,~,~,exitflag] = lsqnonlin(@(x) wmti_watson(x,moments),x0,lb,ub,options);
                
                f(i,j,k) = x(1);
                Da(i,j,k) = x(2);
                Depar(i,j,k) = x(3);
                Deperp(i,j,k) = x(4);
                kappa(i,j,k) = x(5);
                exitflags(i,j,k) = exitflag;
                
            else
                continue
            end
            
            
        end
    end
end

% from kappa, calculate c2, the mean cos^2 of the angle between axons and
% main bundle orientation (an easier metric of orientations dispersion, c2
% varies between 1/3 (isotropic) and 1 (perfectly parallel axons)
Fs = sqrt(pi)./2*exp(-kappa).*erfi(sqrt(kappa));
c2 = 1./(2*sqrt(kappa).*Fs)-1./(2*kappa);

end


function F = wmti_watson(x,moments)

% moments
D0 = moments(1);
D2 = moments(2);
W0 = moments(3);
W2 = moments(4);
W4 = moments(5);

f = x(1);
Da = x(2);
Depar = x(3);
Deperp = x(4);
k = x(5);

dawsonf = 0.5*exp(-k)*sqrt(pi).*erfi(sqrt(k));
p2 = 1/4*(3/(sqrt(k)*dawsonf)-2-3/k);
p4 = 1/(32*k^2)*(105+12*k*(5+k)+(5*sqrt(k)*(2*k-21))/dawsonf);

F(1) = 3*D0 - f*Da - (1-f)*(2*Deperp+Depar);
F(2) = 3/2*D2 - p2*(f*Da + (1-f)*(Depar-Deperp));
F(3) = D2^2+5*D0^2*(1+W0/3) - f*Da^2 - (1-f)*(5*Deperp^2+(Depar-Deperp)^2+10/3*Deperp*(Depar-Deperp));
F(4) = 1/2*D2*(D2+7*D0)+7/12*W2*D0^2 - p2*(f*Da^2+(1-f)*((Depar-Deperp)^2+7/3*Deperp*(Depar-Deperp)));
F(5) = 9*D2^2/4+35/24*W4*D0^2 - p4*(f*Da^2 + (1-f)*(Depar-Deperp)^2);

end