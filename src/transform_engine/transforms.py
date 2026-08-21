import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from src.physical_core.field import PhysicalField

def get_dct_matrix(N: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Evaluate the deterministic basis in float64 before a float32 cast.  Direct float32
    # trigonometry accumulated ~8e-5 round-trip error on a 120x80 regional grid; the cast of
    # the float64 basis reduces that to ~2e-6 and is paid only when a matrix is constructed.
    work_dtype = torch.float64 if dtype == torch.float32 else dtype
    n = torch.arange(N, device=device, dtype=work_dtype).unsqueeze(0)
    k = torch.arange(N, device=device, dtype=work_dtype).unsqueeze(1)
    M = torch.cos(np.pi * k * (n + 0.5) / N)
    M[0, :] *= np.sqrt(1.0 / N)
    M[1:, :] *= np.sqrt(2.0 / N)
    return M.to(dtype=dtype)

class SpectralTransformEngine:
    """
    Computes forward and inverse Fourier, Discrete Cosine, Discrete Wavelet,
    and Dual-Tree Complex Wavelet transforms. Supports hybrid representations.
    """
    
    @staticmethod
    def apply_fft2d(field: PhysicalField) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D Fast Fourier Transform.
        Returns: (magnitude, phase)
        """
        coeffs = torch.fft.rfft2(field.data)
        magnitude = torch.abs(coeffs)
        phase = torch.angle(coeffs)
        return magnitude, phase

    @staticmethod
    def inverse_fft2d(magnitude: torch.Tensor, phase: torch.Tensor, target_shape: Tuple[int, int]) -> PhysicalField:
        """
        Applies 2D Inverse Fast Fourier Transform.
        """
        coeffs = magnitude * torch.exp(1j * phase)
        reconstructed = torch.fft.irfft2(coeffs, s=target_shape)
        return PhysicalField(reconstructed)

    @staticmethod
    def apply_dct2d(field: PhysicalField) -> torch.Tensor:
        """
        Applies 2D Discrete Cosine Transform (Type II).
        """
        data = field.data
        H, W = data.shape
        My = get_dct_matrix(H, data.device, data.dtype)
        Mx = get_dct_matrix(W, data.device, data.dtype)
        coeffs = My @ data @ Mx.T
        return coeffs

    @staticmethod
    def inverse_dct2d(coeffs: torch.Tensor) -> PhysicalField:
        """
        Applies 2D Inverse Discrete Cosine Transform (Type III).
        """
        H, W = coeffs.shape
        My = get_dct_matrix(H, coeffs.device, coeffs.dtype)
        Mx = get_dct_matrix(W, coeffs.device, coeffs.dtype)
        reconstructed = My.T @ coeffs @ Mx
        return PhysicalField(reconstructed)

    @staticmethod
    def _get_haar_filters(
        device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Haar analysis filters.

        dtype must be passed explicitly: ``1.0 / np.sqrt(2.0)`` is a ``numpy.float64``
        scalar, so ``torch.tensor([...])`` infers float64 and conv2d then fails with
        "expected scalar type Float but found Double" against float32 field data (D23).
        """
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        h = torch.tensor([inv_sqrt2, inv_sqrt2], device=device, dtype=dtype)
        g = torch.tensor([inv_sqrt2, -inv_sqrt2], device=device, dtype=dtype)
        return h, g

    @classmethod
    def apply_dwt2d(cls, field: PhysicalField, levels: int = 1) -> Dict[str, Any]:
        """
        Applies multi-level 2D Discrete Wavelet Transform (Haar).
        Returns a dictionary of subbands.
        """
        data = field.data
        h, g = cls._get_haar_filters(data.device, data.dtype)
        
        coeffs = {}
        current = data
        
        for level in range(1, levels + 1):
            H, W = current.shape
            pad_h = H % 2
            pad_w = W % 2
            # torch requires 3D/4D input for a 4-element non-constant pad, so lift to NCHW.
            current_padded = torch.nn.functional.pad(
                current.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0).squeeze(0)
            
            x_input = current_padded.unsqueeze(0).unsqueeze(0)
            
            f_LL = torch.outer(h, h).view(1, 1, 2, 2)
            f_LH = torch.outer(h, g).view(1, 1, 2, 2)
            f_HL = torch.outer(g, h).view(1, 1, 2, 2)
            f_HH = torch.outer(g, g).view(1, 1, 2, 2)
            
            filters = torch.cat([f_LL, f_LH, f_HL, f_HH], dim=0)
            out = torch.nn.functional.conv2d(x_input, filters, stride=2)
            
            LL = out[0, 0]
            LH = out[0, 1]
            HL = out[0, 2]
            HH = out[0, 3]
            
            coeffs[f"level_{level}"] = {"LH": LH, "HL": HL, "HH": HH}
            current = LL
            
        coeffs["LL"] = current
        return coeffs

    @classmethod
    def inverse_dwt2d(cls, coeffs: Dict[str, Any], levels: int = 1, target_shape: Optional[Tuple[int, int]] = None) -> PhysicalField:
        """
        Applies multi-level 2D Inverse Discrete Wavelet Transform.
        """
        device = coeffs["LL"].device
        h, g = cls._get_haar_filters(device, coeffs["LL"].dtype)
        
        current = coeffs["LL"]
        
        for level in range(levels, 0, -1):
            level_coeffs = coeffs[f"level_{level}"]
            LH = level_coeffs["LH"]
            HL = level_coeffs["HL"]
            HH = level_coeffs["HH"]
            
            # D25: stacking 2D tensors on dim=1 yields (H, 4, W); conv_transpose2d
            # needs (N, C, H, W), so stack on dim=0. Also crop the incoming LL, which
            # can exceed this level's detail bands when the forward pass padded an odd size.
            current = current[:LH.shape[0], :LH.shape[1]]
            subbands = torch.stack([current, LH, HL, HH], dim=0).unsqueeze(0)
            
            f_LL = torch.outer(h, h).view(1, 1, 2, 2)
            f_LH = torch.outer(h, g).view(1, 1, 2, 2)
            f_HL = torch.outer(g, h).view(1, 1, 2, 2)
            f_HH = torch.outer(g, g).view(1, 1, 2, 2)
            
            filters = torch.cat([f_LL, f_LH, f_HL, f_HH], dim=0)
            
            out = 0.0
            for i in range(4):
                out += torch.nn.functional.conv_transpose2d(
                    subbands[:, i:i+1], filters[i:i+1], stride=2
                )
            current = out[0, 0]
            
        if target_shape is not None:
            current = current[:target_shape[0], :target_shape[1]]
            
        return PhysicalField(current)

    @classmethod
    def apply_dtcwt2d(cls, field: PhysicalField, levels: int = 1) -> Dict[str, Any]:
        """**DEFECT D1 ARTEFACT - DO NOT USE. Superseded by `transform_engine/dtcwt.py`.**

        This is *not* a dual-tree complex wavelet transform. Tree B's filters are
        ``[cos(pi/4), sin(pi/4)]``, identical to the Haar low-pass, and ``g_b = -g_a``, so
        all four "trees" are one filter bank up to a sign. There is no Hilbert pair, no
        analytic response, no shift invariance and no orientation. It reconstructs perfectly
        because four copies of the same transform average back to the input, which is
        exactly why a round-trip test could never detect the defect (rule R8).

        It is retained, unused by any runtime path, solely as the comparison arm of the
        head-to-head regression tests in `test_dtcwt.py` - the fix must stay demonstrable
        against the thing it fixed. `test_no_runtime_code_uses_the_degenerate_dtcwt`
        enforces that nothing in `src/` outside the tests calls it.

        Measured: 236.11% subband-energy spread over an 0-8 px translation, *identical* to
        a plain Haar DWT on the same input. The real transform scores 4.99%.
        """
        """
        Applies a mathematically rigorous 2D Dual-Tree Complex Wavelet Transform.
        Uses parallel DWT trees (Tree A and Tree B) with orthogonal filters to ensure shift invariance.
        """
        h_a, g_a = cls._get_haar_filters(field.data.device, field.data.dtype)
        theta = np.pi / 4
        # NOTE: these filters are degenerate - they equal tree A up to a sign, which is
        # defect D1. Kept verbatim (only dtype-pinned) until T3.5.6 replaces them with
        # genuine Kingsbury q-shift filters. See architecture.md Section 3.1.
        h_b = torch.tensor([float(np.cos(theta)), float(np.sin(theta))],
                           device=field.data.device, dtype=field.data.dtype)
        g_b = torch.tensor([float(-np.sin(theta)), float(np.cos(theta))],
                           device=field.data.device, dtype=field.data.dtype)
        
        def run_tree(h_row, g_row, h_col, g_col):
            current = field.data
            coeffs = {}
            for level in range(1, levels + 1):
                H, W = current.shape
                pad_h = H % 2
                pad_w = W % 2
                # torch requires 3D/4D input for a 4-element non-constant pad, so lift to NCHW.
                current_padded = torch.nn.functional.pad(
                    current.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
                ).squeeze(0).squeeze(0)
                
                x_input = current_padded.unsqueeze(0).unsqueeze(0)
                
                f_LL = torch.outer(h_row, h_col).view(1, 1, 2, 2)
                f_LH = torch.outer(h_row, g_col).view(1, 1, 2, 2)
                f_HL = torch.outer(g_row, h_col).view(1, 1, 2, 2)
                f_HH = torch.outer(g_row, g_col).view(1, 1, 2, 2)
                
                filters = torch.cat([f_LL, f_LH, f_HL, f_HH], dim=0)
                out = torch.nn.functional.conv2d(x_input, filters, stride=2)
                
                LL = out[0, 0]
                LH = out[0, 1]
                HL = out[0, 2]
                HH = out[0, 3]
                
                coeffs[f"level_{level}"] = {"LH": LH, "HL": HL, "HH": HH}
                current = LL
            coeffs["LL"] = current
            return coeffs

        coeffs_AA = run_tree(h_a, g_a, h_a, g_a)
        coeffs_AB = run_tree(h_a, g_a, h_b, g_b)
        coeffs_BA = run_tree(h_b, g_b, h_a, g_a)
        coeffs_BB = run_tree(h_b, g_b, h_b, g_b)
        
        complex_coeffs = {
            "LL": coeffs_AA["LL"],
            "LL_AA": coeffs_AA["LL"],
            "LL_AB": coeffs_AB["LL"],
            "LL_BA": coeffs_BA["LL"],
            "LL_BB": coeffs_BB["LL"]
        }
        for level in range(1, levels + 1):
            complex_coeffs[f"level_{level}"] = {
                "LH_real": coeffs_AA[f"level_{level}"]["LH"],
                "LH_imag": coeffs_BB[f"level_{level}"]["LH"],
                "HL_real": coeffs_AB[f"level_{level}"]["HL"],
                "HL_imag": coeffs_BA[f"level_{level}"]["HL"],
                "HH_real": coeffs_AA[f"level_{level}"]["HH"],
                "HH_imag": coeffs_BB[f"level_{level}"]["HH"],
                
                "LH_AA": coeffs_AA[f"level_{level}"]["LH"],
                "HL_AA": coeffs_AA[f"level_{level}"]["HL"],
                "HH_AA": coeffs_AA[f"level_{level}"]["HH"],
                
                "LH_AB": coeffs_AB[f"level_{level}"]["LH"],
                "HL_AB": coeffs_AB[f"level_{level}"]["HL"],
                "HH_AB": coeffs_AB[f"level_{level}"]["HH"],
                
                "LH_BA": coeffs_BA[f"level_{level}"]["LH"],
                "HL_BA": coeffs_BA[f"level_{level}"]["HL"],
                "HH_BA": coeffs_BA[f"level_{level}"]["HH"],
                
                "LH_BB": coeffs_BB[f"level_{level}"]["LH"],
                "HL_BB": coeffs_BB[f"level_{level}"]["HL"],
                "HH_BB": coeffs_BB[f"level_{level}"]["HH"]
            }
        return complex_coeffs

    @classmethod
    def inverse_dtcwt2d(cls, coeffs: Dict[str, Any], levels: int = 1, target_shape: Optional[Tuple[int, int]] = None) -> PhysicalField:
        """
        Applies 2D Inverse Dual-Tree Complex Wavelet Transform using all 4 trees for perfect reconstruction.
        """
        device = coeffs["LL"].device
        h_a, g_a = cls._get_haar_filters(device, coeffs["LL"].dtype)
        theta = np.pi / 4
        h_b = torch.tensor([float(np.cos(theta)), float(np.sin(theta))],
                           device=device, dtype=coeffs["LL"].dtype)
        g_b = torch.tensor([float(-np.sin(theta)), float(np.cos(theta))],
                           device=device, dtype=coeffs["LL"].dtype)
        
        def run_inverse_tree(tree_coeffs, h_row, g_row, h_col, g_col):
            current = tree_coeffs["LL"]
            for level in range(levels, 0, -1):
                level_coeffs = tree_coeffs[f"level_{level}"]
                LH = level_coeffs["LH"]
                HL = level_coeffs["HL"]
                HH = level_coeffs["HH"]
                
                # D25: stacking 2D tensors on dim=1 yields (H, 4, W); conv_transpose2d
                # needs (N, C, H, W), so stack on dim=0. Also crop the incoming LL, which
                # can exceed this level's detail bands when the forward pass padded an odd size.
                current = current[:LH.shape[0], :LH.shape[1]]
                subbands = torch.stack([current, LH, HL, HH], dim=0).unsqueeze(0)
                
                f_LL = torch.outer(h_row, h_col).view(1, 1, 2, 2)
                f_LH = torch.outer(h_row, g_col).view(1, 1, 2, 2)
                f_HL = torch.outer(g_row, h_col).view(1, 1, 2, 2)
                f_HH = torch.outer(g_row, g_col).view(1, 1, 2, 2)
                
                filters = torch.cat([f_LL, f_LH, f_HL, f_HH], dim=0)
                
                out = 0.0
                for i in range(4):
                    out += torch.nn.functional.conv_transpose2d(
                        subbands[:, i:i+1], filters[i:i+1], stride=2
                    )
                current = out[0, 0]
            return current

        tree_AA = {"LL": coeffs["LL_AA"]}
        tree_AB = {"LL": coeffs["LL_AB"]}
        tree_BA = {"LL": coeffs["LL_BA"]}
        tree_BB = {"LL": coeffs["LL_BB"]}
        for level in range(1, levels + 1):
            tree_AA[f"level_{level}"] = {
                "LH": coeffs[f"level_{level}"]["LH_AA"],
                "HL": coeffs[f"level_{level}"]["HL_AA"],
                "HH": coeffs[f"level_{level}"]["HH_AA"]
            }
            tree_AB[f"level_{level}"] = {
                "LH": coeffs[f"level_{level}"]["LH_AB"],
                "HL": coeffs[f"level_{level}"]["HL_AB"],
                "HH": coeffs[f"level_{level}"]["HH_AB"]
            }
            tree_BA[f"level_{level}"] = {
                "LH": coeffs[f"level_{level}"]["LH_BA"],
                "HL": coeffs[f"level_{level}"]["HL_BA"],
                "HH": coeffs[f"level_{level}"]["HH_BA"]
            }
            tree_BB[f"level_{level}"] = {
                "LH": coeffs[f"level_{level}"]["LH_BB"],
                "HL": coeffs[f"level_{level}"]["HL_BB"],
                "HH": coeffs[f"level_{level}"]["HH_BB"]
            }
            
        recon_AA = run_inverse_tree(tree_AA, h_a, g_a, h_a, g_a)
        recon_AB = run_inverse_tree(tree_AB, h_a, g_a, h_b, g_b)
        recon_BA = run_inverse_tree(tree_BA, h_b, g_b, h_a, g_a)
        recon_BB = run_inverse_tree(tree_BB, h_b, g_b, h_b, g_b)
        
        reconstructed = 0.25 * (recon_AA + recon_AB + recon_BA + recon_BB)
        
        if target_shape is not None:
            reconstructed = reconstructed[:target_shape[0], :target_shape[1]]
            
        return PhysicalField(reconstructed)

    @classmethod
    def apply_hybrid(cls, field: PhysicalField, crossover_freq: float = 0.5, mixing_weight: float = 0.5) -> Dict[str, Any]:
        """
        Applies a hybrid spectral representation. Vectorized using torch.meshgrid to avoid performance bottlenecks.
        """
        fft_mag, fft_phase = cls.apply_fft2d(field)
        
        H, W = field.data.shape
        y_freq = torch.fft.rfftfreq(W, d=1.0, device=field.data.device)
        x_freq = torch.fft.fftfreq(H, d=1.0, device=field.data.device)
        grid_y, grid_x = torch.meshgrid(x_freq, y_freq, indexing="ij")
        dist = torch.sqrt(grid_y**2 + grid_x**2)
        
        mask = (dist <= crossover_freq).to(field.data.device)
        fft_mag_filtered = fft_mag * mask
        
        low_freq_field = cls.inverse_fft2d(fft_mag_filtered, fft_phase, (H, W))
        
        residual_data = field.data - low_freq_field.data
        residual_field = PhysicalField(residual_data)
        dwt_coeffs = cls.apply_dwt2d(residual_field, levels=1)
        
        return {
            "fft_mag": fft_mag,
            "fft_phase": fft_phase,
            "fft_mag_filtered": fft_mag_filtered,
            "dwt_coeffs": dwt_coeffs,
            "mixing_weight": mixing_weight,
            "crossover_freq": crossover_freq
        }

    @classmethod
    def inverse_hybrid(
        cls,
        hybrid_coeffs: Dict[str, Any],
        target_shape: Tuple[int, int],
        exact: bool = True,
    ) -> PhysicalField:
        """
        Reconstructs the field from the hybrid representation.

        The forward pass decomposes the field as ``field = low + residual``, so the
        inverse is the *sum* of the two components. Defect D2 was that this method
        instead returned ``w * low + (1 - w) * residual``, which is not an inverse for
        any value of ``w`` -- the ``hybrid`` transform therefore always reported a
        non-zero reconstruction error for algebraic rather than physical reasons.

        exact=True  : faithful inverse, ``low + residual`` (default).
        exact=False : deliberately lossy blend using ``mixing_weight``, retained only
                      for studying the effect of down-weighting one component. This is
                      NOT a reconstruction and its error must not be read as one.
        """
        fft_mag_filtered = hybrid_coeffs["fft_mag_filtered"]
        fft_phase = hybrid_coeffs["fft_phase"]
        dwt_coeffs = hybrid_coeffs["dwt_coeffs"]

        low_freq_field = cls.inverse_fft2d(fft_mag_filtered, fft_phase, target_shape)
        high_freq_field = cls.inverse_dwt2d(dwt_coeffs, levels=1)

        H, W = target_shape
        high_freq_data = high_freq_field.data[:H, :W]

        if exact:
            combined_data = low_freq_field.data + high_freq_data
        else:
            mixing_weight = hybrid_coeffs["mixing_weight"]
            combined_data = (
                2.0 * mixing_weight * low_freq_field.data
                + 2.0 * (1.0 - mixing_weight) * high_freq_data
            )
        return PhysicalField(combined_data)
