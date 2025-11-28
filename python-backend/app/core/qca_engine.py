"""
Quantitative Coronary Analysis (QCA) Engine

Calculates vessel diameter profiles and stenosis metrics from
segmentation masks along the centerline with sub-pixel accuracy.

Features:
- N-point diameter profiling (30/50/70 points)
- Perpendicular cross-section sampling
- Sub-pixel diameter with Gaussian/Parabolic fitting
- Bilinear interpolation for accurate sampling
- Arc-length based centerline resampling

Metrics:
- MLD (Minimum Lumen Diameter)
- Proximal Reference Diameter
- Distal Reference Diameter
- Diameter Stenosis (DS%)
- Lesion Length (LL)
- Interpolated Reference Diameter

References:
- Gould, K.L. et al., "Coronary flow reserve as a physiologic measure
  of stenosis severity", JACC 1990
- SCAI Guidelines for Quantitative Coronary Analysis
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    from scipy import ndimage
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QCAMetrics:
    """
    QCA metrics for a single frame.

    All diameter values are in the unit specified by pixel_spacing.
    If pixel_spacing is in mm/pixel, diameters are in mm.
    """
    # Diameter profile
    diameter_profile: List[float]  # N-point diameter array (mm)
    diameter_profile_px: List[float]  # N-point diameter array (pixels)
    centerline_points: List[Tuple[float, float]]  # Corresponding (y, x) points
    distances_mm: List[float]  # Arc-length distances along centerline

    # Key measurements
    mld: float  # Minimum Lumen Diameter (mm)
    mld_index: int  # Index of MLD in profile
    mld_position: Tuple[float, float]  # (y, x) position of MLD

    proximal_rd: float  # Proximal Reference Diameter (mm)
    proximal_rd_index: int
    proximal_rd_position: Tuple[float, float]

    distal_rd: float  # Distal Reference Diameter (mm)
    distal_rd_index: int
    distal_rd_position: Tuple[float, float]

    # Derived metrics
    interpolated_rd: float  # Linear interpolation between Proximal and Distal RD at MLD
    diameter_stenosis: float  # DS% = (1 - MLD / Int_RD) × 100
    lesion_length: Optional[float]  # Distance where diameter < threshold (mm)

    # Metadata
    pixel_spacing: float  # mm/pixel used for conversion
    num_points: int  # N in N-point analysis
    method: str  # Diameter calculation method used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "diameter_profile": self.diameter_profile,
            "diameter_profile_px": self.diameter_profile_px,
            "centerline_points": [(float(y), float(x)) for y, x in self.centerline_points],
            "centerline_with_diameters": [
                {"x": float(x), "y": float(y), "diameter": float(d)}
                for (y, x), d in zip(self.centerline_points, self.diameter_profile)
            ],
            "distances_mm": self.distances_mm,
            "mld": self.mld,
            "mld_mm": self.mld,  # Alias for compatibility
            "mld_index": self.mld_index,
            "mld_position": list(self.mld_position),
            "proximal_rd": self.proximal_rd,
            "proximal_rd_mm": self.proximal_rd,  # Alias
            "proximal_rd_index": self.proximal_rd_index,
            "proximal_rd_position": list(self.proximal_rd_position),
            "distal_rd": self.distal_rd,
            "distal_rd_mm": self.distal_rd,  # Alias
            "distal_rd_index": self.distal_rd_index,
            "distal_rd_position": list(self.distal_rd_position),
            "interpolated_rd": self.interpolated_rd,
            "interpolated_reference_mm": self.interpolated_rd,  # Alias
            "diameter_stenosis": self.diameter_stenosis,
            "diameter_stenosis_percent": self.diameter_stenosis,  # Alias
            "lesion_length": self.lesion_length,
            "lesion_length_mm": self.lesion_length,  # Alias
            "pixel_spacing": self.pixel_spacing,
            "num_points": self.num_points,
            "n_points": self.num_points,  # Alias
            "method": self.method,
            "mean_diameter_mm": float(np.mean(self.diameter_profile)) if self.diameter_profile else 0.0,
            "std_diameter_mm": float(np.std(self.diameter_profile)) if self.diameter_profile else 0.0,
        }


class QCAEngine:
    """
    QCA calculation engine with sub-pixel accuracy.

    Calculates diameter profiles and stenosis metrics from segmentation
    masks using perpendicular measurement along the centerline.

    Supports multiple diameter fitting methods:
    - gaussian: Gaussian fitting for sub-pixel FWHM measurement
    - parabolic: Parabolic fitting for sub-pixel accuracy
    - threshold: Simple threshold-based measurement (fastest)

    Usage:
        engine = QCAEngine(num_points=50, method='gaussian')
        metrics = engine.calculate(mask, centerline, pixel_spacing=0.3)

        print(f"MLD: {metrics.mld:.2f} mm")
        print(f"DS%: {metrics.diameter_stenosis:.1f}%")
    """

    def __init__(self, num_points: int = 50, method: str = "gaussian"):
        """
        Initialize QCA engine.

        Args:
            num_points: Number of measurement points along centerline (30/50/70)
            method: Diameter calculation method ('gaussian', 'parabolic', 'threshold')
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for QCA calculations")

        self.num_points = num_points
        self.method = method
        self._last_metrics: Optional[QCAMetrics] = None

    def calculate(
        self,
        mask: np.ndarray,
        centerline: np.ndarray,
        pixel_spacing: float = 1.0,
        probability_map: Optional[np.ndarray] = None
    ) -> Optional[QCAMetrics]:
        """
        Calculate QCA metrics from segmentation mask.

        Args:
            mask: Binary segmentation mask (non-zero = vessel)
            centerline: Centerline points as (N, 2) array of (y, x)
            pixel_spacing: mm/pixel conversion factor (from DICOM)
            probability_map: Optional probability map for sub-pixel accuracy

        Returns:
            QCAMetrics with all calculated values
        """
        if mask is None or len(mask.shape) != 2:
            logger.error("Invalid mask")
            return None

        if centerline is None or len(centerline) < 3:
            logger.error("Invalid centerline (need at least 3 points)")
            return None

        try:
            # Normalize centerline format
            centerline = self._normalize_centerline(centerline)

            # Resample centerline to N points with equal arc-length spacing
            sampled_centerline = self._resample_centerline(centerline, self.num_points)

            # Use probability map if available, else use mask
            if probability_map is None:
                probability_map = mask.astype(np.float32)

            # Step 1: Compute diameter profile with sub-pixel accuracy
            diameter_profile_px = self._compute_diameter_profile(
                mask, probability_map, sampled_centerline
            )

            if len(diameter_profile_px) == 0:
                logger.error("Failed to compute diameter profile")
                return None

            # Convert to mm
            diameter_profile_mm = [d * pixel_spacing for d in diameter_profile_px]

            # Calculate distances along centerline
            distances_mm = self._calculate_distances(sampled_centerline, pixel_spacing)

            # Step 2: Find MLD
            mld_idx = int(np.argmin(diameter_profile_mm))
            mld = diameter_profile_mm[mld_idx]
            mld_pos = tuple(sampled_centerline[mld_idx])

            # Step 3: Find Reference Diameters
            ref_window = max(1, len(diameter_profile_mm) // 5)

            # Proximal: maximum in first N/5 points
            proximal_region = diameter_profile_mm[:ref_window]
            proximal_idx = int(np.argmax(proximal_region))
            proximal_rd = proximal_region[proximal_idx]
            proximal_pos = tuple(sampled_centerline[proximal_idx])

            # Distal: maximum in last N/5 points
            distal_region = diameter_profile_mm[-ref_window:]
            distal_rel_idx = int(np.argmax(distal_region))
            distal_idx = len(diameter_profile_mm) - ref_window + distal_rel_idx
            distal_rd = diameter_profile_mm[distal_idx]
            distal_pos = tuple(sampled_centerline[distal_idx])

            # Step 4: Calculate Interpolated Reference Diameter at MLD position
            if distal_idx != proximal_idx:
                t = (mld_idx - proximal_idx) / (distal_idx - proximal_idx)
                interpolated_rd = proximal_rd + t * (distal_rd - proximal_rd)
            else:
                interpolated_rd = (proximal_rd + distal_rd) / 2

            # Step 5: Calculate Diameter Stenosis
            if interpolated_rd > 0:
                ds_percent = (1 - mld / interpolated_rd) * 100
                ds_percent = max(0.0, min(100.0, ds_percent))  # Clamp
            else:
                ds_percent = 0.0

            # Step 6: Calculate Lesion Length
            lesion_length = self._calculate_lesion_length(
                sampled_centerline, diameter_profile_mm, interpolated_rd, pixel_spacing
            )

            # Build result
            metrics = QCAMetrics(
                diameter_profile=diameter_profile_mm,
                diameter_profile_px=diameter_profile_px,
                centerline_points=[(float(y), float(x)) for y, x in sampled_centerline],
                distances_mm=distances_mm,
                mld=mld,
                mld_index=mld_idx,
                mld_position=mld_pos,
                proximal_rd=proximal_rd,
                proximal_rd_index=proximal_idx,
                proximal_rd_position=proximal_pos,
                distal_rd=distal_rd,
                distal_rd_index=distal_idx,
                distal_rd_position=distal_pos,
                interpolated_rd=interpolated_rd,
                diameter_stenosis=ds_percent,
                lesion_length=lesion_length,
                pixel_spacing=pixel_spacing,
                num_points=len(diameter_profile_mm),
                method=self.method
            )

            self._last_metrics = metrics

            logger.info(
                f"QCA ({self.method}): MLD={mld:.2f}mm, Prox_RD={proximal_rd:.2f}mm, "
                f"Dist_RD={distal_rd:.2f}mm, DS={ds_percent:.1f}%"
            )

            return metrics

        except Exception as e:
            logger.error(f"QCA calculation failed: {e}")
            return None

    def _normalize_centerline(self, centerline: np.ndarray) -> np.ndarray:
        """Ensure centerline is in (N, 2) format as (y, x)."""
        if centerline.ndim != 2 or centerline.shape[1] != 2:
            raise ValueError(f"Invalid centerline shape: {centerline.shape}")
        return centerline.astype(np.float32)

    def _resample_centerline(
        self,
        centerline: np.ndarray,
        n_points: int
    ) -> np.ndarray:
        """
        Resample centerline to exactly N points with equal arc-length spacing.

        Args:
            centerline: (M, 2) array as (y, x)
            n_points: Target number of points

        Returns:
            (N, 2) resampled centerline
        """
        if len(centerline) <= 1:
            return centerline

        # Calculate cumulative arc length
        diffs = np.diff(centerline, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length == 0:
            return np.tile(centerline[0], (n_points, 1))

        # Target arc lengths for N points
        target_lengths = np.linspace(0, total_length, n_points)

        # Interpolate y and x separately
        interp_y = interp1d(cumulative_length, centerline[:, 0], kind='linear')
        interp_x = interp1d(cumulative_length, centerline[:, 1], kind='linear')

        resampled_y = interp_y(target_lengths)
        resampled_x = interp_x(target_lengths)

        return np.column_stack([resampled_y, resampled_x])

    def _compute_diameter_profile(
        self,
        mask: np.ndarray,
        probability_map: np.ndarray,
        centerline: np.ndarray
    ) -> List[float]:
        """
        Compute diameter at each centerline point using perpendicular cross-sections.

        Uses sub-pixel fitting for accurate measurements.

        Args:
            mask: Binary segmentation mask
            probability_map: Probability values for sub-pixel accuracy
            centerline: Sampled centerline points (N, 2) as (y, x)

        Returns:
            List of diameter values in pixels
        """
        diameters = []

        for i, (cy, cx) in enumerate(centerline):
            # Calculate tangent direction
            tangent = self._calculate_tangent(centerline, i)

            # Perpendicular direction
            normal = np.array([-tangent[1], tangent[0]])

            # Measure diameter along perpendicular
            diameter = self._measure_diameter_at_point(
                probability_map,
                (cy, cx),
                normal,
                method=self.method
            )

            diameters.append(diameter)

        return diameters

    def _calculate_tangent(
        self,
        centerline: np.ndarray,
        idx: int
    ) -> np.ndarray:
        """
        Calculate tangent vector at centerline point using finite differences.
        """
        n_points = len(centerline)

        if idx == 0:
            # Forward difference
            tangent = centerline[idx + 1] - centerline[idx]
        elif idx == n_points - 1:
            # Backward difference
            tangent = centerline[idx] - centerline[idx - 1]
        else:
            # Central difference (more accurate)
            tangent = centerline[idx + 1] - centerline[idx - 1]

        # Normalize
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])

        return tangent

    def _measure_diameter_at_point(
        self,
        probability_map: np.ndarray,
        center: Tuple[float, float],
        normal: np.ndarray,
        method: str = "gaussian",
        max_radius: int = 50
    ) -> float:
        """
        Measure diameter along perpendicular line using sub-pixel fitting.

        Args:
            probability_map: Probability values
            center: (y, x) center point
            normal: (dy, dx) perpendicular direction (normalized)
            method: Fitting method ('gaussian', 'parabolic', 'threshold')
            max_radius: Maximum search radius

        Returns:
            Diameter in pixels
        """
        cy, cx = center
        ny, nx = normal

        # Sample points along perpendicular (both directions)
        t_values = np.linspace(-max_radius, max_radius, 2 * max_radius + 1)
        sample_y = cy + t_values * ny
        sample_x = cx + t_values * nx

        # Clip to image bounds
        h, w = probability_map.shape
        valid_mask = (
            (sample_y >= 0) & (sample_y < h - 1) &
            (sample_x >= 0) & (sample_x < w - 1)
        )

        if not valid_mask.any():
            return 0.0

        # Bilinear interpolation for sub-pixel sampling
        profile = self._bilinear_sample(
            probability_map,
            sample_y[valid_mask],
            sample_x[valid_mask]
        )

        t_valid = t_values[valid_mask]

        # Detect vessel edges based on method
        if method == "gaussian":
            diameter = self._fit_gaussian_diameter(t_valid, profile)
        elif method == "parabolic":
            diameter = self._fit_parabolic_diameter(t_valid, profile)
        else:  # threshold
            diameter = self._threshold_diameter(t_valid, profile)

        return diameter

    def _bilinear_sample(
        self,
        image: np.ndarray,
        y_coords: np.ndarray,
        x_coords: np.ndarray
    ) -> np.ndarray:
        """Bilinear interpolation for sub-pixel sampling."""
        y0 = np.floor(y_coords).astype(int)
        x0 = np.floor(x_coords).astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        # Clip to image bounds
        h, w = image.shape
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)

        # Fractional parts
        fy = y_coords - np.floor(y_coords)
        fx = x_coords - np.floor(x_coords)

        # Bilinear weights
        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy

        # Interpolate
        values = (
            w00 * image[y0, x0] +
            w01 * image[y1, x0] +
            w10 * image[y0, x1] +
            w11 * image[y1, x1]
        )

        return values

    def _fit_gaussian_diameter(
        self,
        t: np.ndarray,
        profile: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Fit Gaussian to intensity profile and measure FWHM.

        Gaussian model: f(t) = A * exp(-((t - mu)^2) / (2 * sigma^2)) + offset
        Diameter ≈ 2.355 * sigma (FWHM)
        """
        if len(profile) < 5:
            return self._threshold_diameter(t, profile, threshold)

        # Initial guess
        A_init = profile.max() - profile.min()
        mu_init = t[np.argmax(profile)]
        sigma_init = 5.0
        offset_init = profile.min()

        def gaussian(x, A, mu, sigma, offset):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

        try:
            popt, _ = curve_fit(
                gaussian,
                t,
                profile,
                p0=[A_init, mu_init, sigma_init, offset_init],
                maxfev=1000
            )

            A, mu, sigma, offset = popt

            # FWHM = 2.355 * sigma
            diameter = 2.355 * abs(sigma)

            # Sanity check
            if diameter > len(t) or diameter < 1:
                return self._threshold_diameter(t, profile, threshold)

            return diameter

        except Exception as e:
            logger.debug(f"Gaussian fitting failed: {e}, using threshold method")
            return self._threshold_diameter(t, profile, threshold)

    def _fit_parabolic_diameter(
        self,
        t: np.ndarray,
        profile: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Fit parabola to intensity profile.

        Parabolic model: f(t) = a * (t - mu)^2 + peak
        """
        if len(profile) < 5:
            return self._threshold_diameter(t, profile, threshold)

        # Find peak
        peak_idx = np.argmax(profile)
        peak_t = t[peak_idx]
        peak_val = profile[peak_idx]

        def parabola(x, a, mu, peak):
            return a * (x - mu) ** 2 + peak

        try:
            popt, _ = curve_fit(
                parabola,
                t,
                profile,
                p0=[-0.01, peak_t, peak_val],
                maxfev=1000
            )

            a, mu, peak = popt

            if a >= 0:  # Invalid (should be negative for vessel)
                return self._threshold_diameter(t, profile, threshold)

            threshold_val = threshold * peak
            delta_sq = (threshold_val - peak) / a

            if delta_sq < 0:
                return self._threshold_diameter(t, profile, threshold)

            delta = np.sqrt(delta_sq)
            diameter = 2 * delta

            # Sanity check
            if diameter > len(t) or diameter < 1:
                return self._threshold_diameter(t, profile, threshold)

            return diameter

        except Exception as e:
            logger.debug(f"Parabolic fitting failed: {e}, using threshold method")
            return self._threshold_diameter(t, profile, threshold)

    def _threshold_diameter(
        self,
        t: np.ndarray,
        profile: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Simple threshold-based diameter measurement.

        Find where profile crosses threshold * max_value
        """
        if len(profile) == 0:
            return 0.0

        max_val = profile.max()
        threshold_val = threshold * max_val

        # Find crossings
        above_threshold = profile >= threshold_val

        if not above_threshold.any():
            return 0.0

        # Find first and last crossing
        indices = np.where(above_threshold)[0]
        first_idx = indices[0]
        last_idx = indices[-1]

        # Diameter = distance between crossings
        diameter = abs(t[last_idx] - t[first_idx])

        return diameter

    def _calculate_distances(
        self,
        centerline: np.ndarray,
        pixel_spacing: float
    ) -> List[float]:
        """Calculate cumulative arc-length distances along centerline (mm)."""
        diffs = np.diff(centerline, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        distances_px = np.concatenate([[0], np.cumsum(segment_lengths)])
        distances_mm = (distances_px * pixel_spacing).tolist()
        return distances_mm

    def _calculate_lesion_length(
        self,
        centerline: np.ndarray,
        diameter_profile: List[float],
        interpolated_ref: float,
        pixel_spacing: float,
        threshold: float = 0.5
    ) -> Optional[float]:
        """
        Calculate lesion length where diameter is below threshold.

        Args:
            centerline: Sampled centerline points (y, x)
            diameter_profile: Diameter at each point (mm)
            interpolated_ref: Interpolated reference diameter (mm)
            pixel_spacing: mm/pixel conversion
            threshold: Fraction of reference diameter for lesion definition

        Returns:
            Lesion length in mm, or None if no lesion
        """
        try:
            threshold_diameter = threshold * interpolated_ref
            lesion_mask = np.array(diameter_profile) < threshold_diameter

            if not np.any(lesion_mask):
                return None

            # Find start and end of lesion
            lesion_indices = np.where(lesion_mask)[0]
            start_idx = lesion_indices[0]
            end_idx = lesion_indices[-1]

            if start_idx == end_idx:
                return 0.0

            # Calculate arc length along centerline
            segment = centerline[start_idx:end_idx + 1]
            diffs = np.diff(segment, axis=0)
            segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
            lesion_length_px = segment_lengths.sum()

            lesion_length_mm = lesion_length_px * pixel_spacing

            return lesion_length_mm

        except Exception as e:
            logger.debug(f"Lesion length calculation failed: {e}")
            return None

    def get_diameter_at_position(
        self,
        mask: np.ndarray,
        position: Tuple[float, float],
        pixel_spacing: float = 1.0
    ) -> float:
        """
        Get vessel diameter at a specific position.

        Args:
            mask: Binary segmentation mask
            position: (y, x) position to measure
            pixel_spacing: mm/pixel conversion

        Returns:
            Diameter in mm
        """
        try:
            binary_mask = (mask > 0).astype(np.uint8)
            edt = ndimage.distance_transform_edt(binary_mask)

            y, x = int(position[0]), int(position[1])

            if 0 <= y < edt.shape[0] and 0 <= x < edt.shape[1]:
                radius = edt[y, x]
                return radius * 2 * pixel_spacing

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get diameter: {e}")
            return 0.0

    def get_last_metrics(self) -> Optional[QCAMetrics]:
        """Get last calculated metrics."""
        return self._last_metrics


# Singleton instance
_engine_instance: Optional[QCAEngine] = None


def get_engine(num_points: int = 50, method: str = "gaussian") -> QCAEngine:
    """Get singleton QCAEngine instance."""
    global _engine_instance
    if _engine_instance is None or _engine_instance.num_points != num_points or _engine_instance.method != method:
        _engine_instance = QCAEngine(num_points=num_points, method=method)
    return _engine_instance
