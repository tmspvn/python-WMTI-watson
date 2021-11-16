import numpy as np
import nibabel as nib
from scipy.special import erfi
from scipy.optimize import least_squares
from multiprocessing import Pool
import time


# %% Class
class WMTI_Watson:
    def __init__(self, inpath, mask=None, invivo=True, nodes=2, rand=False):
        self.invivo = invivo
        self.rand = rand
        self.nodes = nodes
        self.md = nib.load(inpath + '/md.nii')
        self.md_affine = nib.load(inpath + '/md.nii').affine
        self.md_header = nib.load(inpath + '/md.nii').header
        self.ad = nib.load(inpath + '/ad.nii')
        self.rd = nib.load(inpath + '/rd.nii')
        self.mk = nib.load(inpath + '/mk.nii')
        self.ak = nib.load(inpath + '/ak.nii')
        self.rk = nib.load(inpath + '/rk.nii')
        # load ROI mask
        if mask is None:
            self.mask = np.logical_not(np.isnan(self.md.get_fdata()))
        else:
            self.mask = nib.load(mask).get_fdata()
        if np.logical_not(
                self.md.shape == self.ad.shape == self.rd.shape == self.mk.shape == self.ak.shape == self.rk.shape == self.mask.shape):
            raise ValueError('Inputs shapes are not consistent. Volumes must have the same shape')

    def version(self):
        return print('v1-16.11.21')

    def fit(self):
        t = time.time()
        self.f, self.Da, self.Depar, self.Deperp, self.c2 = WMTI_Watson_maps(self.md.get_fdata(), self.ad.get_fdata(),
                                                                             self.rd.get_fdata(), self.mk.get_fdata(),
                                                                             self.ak.get_fdata(), self.rk.get_fdata(),
                                                                             self.mask, invivo_flag=self.invivo,
                                                                             rand=self.rand, nodes=self.nodes)
        return print(np.round_(time.time() - t, 3), 's')

    def save(self, outpath):
        for out in ['f', 'Da', 'Depar', 'Deperp', 'c2']:
            newimg = nib.Nifti1Image(eval('self.' + out), affine=self.md_affine, header=self.md_header)
            nib.save(newimg, outpath + '/' + out + '.nii')
        return


# %% Functions
def wmti_watson_f(x, moments, rand=True):

    # moments
    D0 = moments[0]
    D2 = moments[1]
    W0 = moments[2]
    W2 = moments[3]
    W4 = moments[4]

    f = x[0]
    Da = x[1]
    Depar = x[2]
    Deperp = x[3]
    k = x[4]

    dawsonf = 0.5 * np.exp(-k) * np.sqrt(np.pi) * erfi(np.sqrt(k))
    p2 = 1 / 4 * (3 / (np.sqrt(k) * dawsonf) - 2 - 3 / k)
    p4 = 1 / (32 * k ** 2) * (105 + 12 * k * (5 + k) + (5 * np.sqrt(k) * (2 * k - 21)) / dawsonf)

    F1 = 3 * D0 - f * Da - (1 - f) * (2 * Deperp + Depar)
    F2 = 3 / 2 * D2 - p2 * (f * Da + (1 - f) * (Depar - Deperp))
    F3 = D2 ** 2 + 5 * D0 ** 2 * (1 + W0 / 3) - f * Da ** 2 - (1 - f) * (
                5 * Deperp ** 2 + (Depar - Deperp) ** 2 + 10 / 3 * Deperp * (Depar - Deperp))
    F4 = 1 / 2 * D2 * (D2 + 7 * D0) + 7 / 12 * W2 * D0 ** 2 - p2 * (
                f * Da ** 2 + (1 - f) * ((Depar - Deperp) ** 2 + 7 / 3 * Deperp * (Depar - Deperp)))
    F5 = 9 * D2 ** 2 / 4 + 35 / 24 * W4 * D0 ** 2 - p4 * (f * Da ** 2 + (1 - f) * (Depar - Deperp) ** 2)

    return np.array([F1, F2, F3, F4, F5])


def normal_fit_wmti_watson(roi, D0, D2, W0, W2, W4, x0, lb, ub, rand=False):
    # empty storage
    fx0, fx1, fx2, fx3, fx4 = [], [], [], [], []
    for i in range(roi[roi].flatten().shape[0]):
        print(str(i) + '/' + str(roi[roi].flatten().shape[0] - 1))


        moments = np.array([D0[roi][i], D2[roi][i], W0[roi][i], W2[roi][i], W4[roi][i]])
        F = least_squares(wmti_watson_f, x0=np.array(x0), bounds=(lb, ub), args=[moments],
                          ftol=1e-6)  # 6 on matlab, ftol

        fx0 += [F.x[0]]
        fx1 += [F.x[1]]
        fx2 += [F.x[2]]
        fx3 += [F.x[3]]
        fx4 += [F.x[4]]
        return fx0, fx1, fx2, fx3, fx4


def parfit_wmti_watson(D0, D2, W0, W2, W4, x0, lb, ub):
        moments = [D0, D2, W0, W2, W4]
        F = least_squares(wmti_watson_f, x0=np.array(x0), bounds=(lb, ub), args=[moments],
                          ftol=1e-6)  # 6 on matlab, ftol
        fx0 = [F.x[0]]
        fx1 = [F.x[1]]
        fx2 = [F.x[2]]
        fx3 = [F.x[3]]
        fx4 = [F.x[4]]

        return fx0, fx1, fx2, fx3, fx4

def rand_x0(len):
    return np.array([np.random.uniform(0.1, 0.8, len),
                    np.random.uniform(1.5, 2.9, len),
                    np.random.uniform(1.0, 1.8, len),
                    np.random.uniform(0.1, 2.5, len),
                    np.random.uniform(1 / 3, 1, len)]).T


def WMTI_Watson_maps(md, ad, rd, mk, ak, rk, mask=None, invivo_flag=True, rand=False, nodes=2):
    '''
    # given md, ad, rd, mk, ak, rk(mean, axial, radial diffusivity, mean, axial, radial kurtosis) maps,
    # calculate WM model parameter maps:
    # f(axonal water fraction), Da(axonal diffusivity), Depar, Deperp(extra - axonal)
    # parallel and perpendicular diffusivities), c2(mean cos ^ 2 of the axon)
    # orientation dispersion: c2 = 1 / 3 fully isotropic, c2 = 1 perfectly parallel)
    # c2 is directly related to the Watson distribution concentration parameter
    # kappa(same as in NODDI)

    # All diffusivities in um2 / ms
    # md, ad, rd should also be in um2 / ms, otherwise converted here

    # mask: brain or ROI mask
    # invivo_flag: boolean, flag for in vivo(true) or ex vivo (false)

    # I.Jelescu, Tommaso Pavan, Nov. 2021
    '''

    # avoid /0
    small = 0.0000000001
    # fit lower bound for model parameters[f, Da, Depar, Deperp, kappa]
    lb = [0, 0, 0, 0, 0]
    if invivo_flag:
        ub = [1, 3, 3, 3, 128]  # fit upper bound for model parameters, in vivo
        x0 = [0.9, 2.2, 1.6, 0.7, 7]  # initial guess, in vivo ### [f, Da, Depar, Deperp, c2]
        md_ub = 2.5
    else:
        ub = [1, 2, 2, 2, 128]  # fit upper bound for model parameters, ex vivo
        x0 = [0.9, 1.6, 1, 0.4, 7]  # initial guess, ex vivo
        md_ub = 1.8  # upper bound on md to avoid CSF contamination

    # Check md in um2 / ms and not mm2 / s, if not, convert
    if np.nanmax(md) < 1e-2:
        md = md * 1e3
        ad = ad * 1e3
        rd = rd * 1e3

    # Make mask #redundant
    if mask is None:
        mask = np.logical_not(np.isnan(md))

    # filter out voxels with unrealistic tensor values
    filt = (md < md_ub) & (rk > 0) & (rk < 10) & (mk > 0) & (mk < 10)
    roi = mask & filt  # exclude voxels with unphysical values from calculation

    # calculate signal moments
    Wpar = ak * (ad / md) ** 2
    Wperp = rk * (rd / md) ** 2

    D0 = md  # mean diffusivity
    D2 = 2 / 3 * (ad - rd)
    W0 = mk
    W2 = 1 / 7 * (3 * Wpar + 5 * mk - 8 * Wperp)
    W4 = 4 / 7 * (Wpar - 3 * mk + 2 * Wperp)

    # random starting points?
    if rand:
        prep_x0 = rand_x0(D0[roi].shape[0])
    else:
        prep_x0 = np.tile(x0, (D0[roi].shape[0], 1))

    # initialize model parameter maps
    f = np.zeros_like(md)
    Da = f.copy()
    Depar = f.copy()
    Deperp = f.copy()
    kappa = f.copy()

    # Fit
    # Prepare parameters for parallelization
    zipped = list(zip(D0[roi], D2[roi], W0[roi], W2[roi], W4[roi],
                      prep_x0,
                      np.tile(lb, (D0[roi].shape[0], 1)),
                      np.tile(ub, (D0[roi].shape[0], 1))))
    p = Pool(nodes)
    mapped = np.array(p.starmap(parfit_wmti_watson, zipped)).squeeze()

    # Store data in place
    f[roi] = mapped[:, 0]
    Da[roi] = mapped[:, 1]
    Depar[roi] = mapped[:, 2]
    Deperp[roi] = mapped[:, 3]
    kappa[roi] = mapped[:, 4]

    # from kappa, calculate c2, the mean cos ^ 2 of the angle between axons and
    # main bundle orientation(an easier metric of orientations dispersion, c2
    # varies between 1 / 3(isotropic) and 1(perfectly parallel axons)
    Fs = np.sqrt(np.pi) / 2 * np.exp(-kappa) * erfi(np.sqrt(kappa))
    c2 = 1 / (2 * np.sqrt(kappa) * Fs + small) - 1 / (2 * kappa + small)

    return f, Da, Depar, Deperp, np.array(c2)


































