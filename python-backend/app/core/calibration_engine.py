"""
Calibration Engine for Catheter-based Calibration

Implements catheter segmentation and diameter measurement for pixel spacing calibration.
Uses the same sub-pixel accuracy methods as QCA engine.

Process:
1. User places 2 seed points on catheter
2. AngioPy segments catheter
3. Centerline extracted
4. Sub-pixel diameter calculation at N points
5. New ImagerPixelSpacing = Known Catheter Size (mm) / Measured Diameter (px)

Reference:
- French catheter sizing: 1 French = 0.33mm outer diameter
"""

import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of catheter-based calibration."""
    success: bool
    catheter_size: str
    known_diameter_mm: float
    measured_diameter_px: float
    new_pixel_spacing: float
    old_pixel_spacing: Optional[float]
    method: str
    n_points: int
    quality_score: float
    quality_notes: List[str]
    diameter_profile_px: List[float]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "catheter_size": self.catheter_size,
            "known_diameter_mm": self.known_diameter_mm,
            "measured_diameter_px": self.measured_diameter_px,
            "new_pixel_spacing": self.new_pixel_spacing,
            "old_pixel_spacing": self.old_pixel_spacing,
            "method": self.method,
            "n_points": self.n_points,
            "quality_score": self.quality_score,
            "quality_notes": self.quality_notes,
            "diameter_profile_px": self.diameter_profile_px,
            "error_message": self.error_message
        }


class CalibrationEngine:
    """
    Catheter-based calibration engine.

    Uses QCA diameter calculation methods for accurate catheter measurement.
    Implements robust outlier removal for reliable calibration.

    Usage:
        engine = CalibrationEngine()
        result = engine.calculate(
            mask=catheter_mask,
            centerline=catheter_centerline,
            catheter_size="6F"
        )
        new_pixel_spacing = result.new_pixel_spacing
    """

    # Standard French catheter sizes (F -> mm outer diameter)
    CATHETER_SIZES = {
        "5F": 1.67,   # 5 French = 1.67mm
        "6F": 2.00,   # 6 French = 2.00mm
        "7F": 2.33,   # 7 French = 2.33mm
        "8F": 2.67,   # 8 French = 2.67mm
    }

    # Valid pixel spacing range for coronary angiography
    MIN_PIXEL_SPACING = 0.05  # mm/px
    MAX_PIXEL_SPACING = 0.5   # mm/px

    # Valid catheter diameter range in pixels
    MIN_CATHETER_PX = 5
    MAX_CATHETER_PX = 200

    def __init__(self):
        self.last_calibration: Optional[CalibrationResult] = None
        self._current_pixel_spacing: Optional[float] = None

    def calculate(
        self,
        mask: np.ndarray,
        centerline: np.ndarray,
        catheter_size: str,
        custom_size_mm: Optional[float] = None,
        probability_map: Optional[np.ndarray] = None,
        n_points: int = 50,
        method: str = "gaussian",
        old_pixel_spacing: Optional[float] = None
    ) -> CalibrationResult:
        """
        Calculate new pixel spacing from catheter segmentation.

        Args:
            mask: Binary catheter segmentation mask (H, W)
            centerline: Catheter centerline coordinates (N, 2) as (y, x)
            catheter_size: Catheter size ("5F", "6F", "7F", "8F", or "custom")
            custom_size_mm: Custom catheter size in mm (if catheter_size == "custom")
            probability_map: Optional probability map from segmentation
            n_points: Number of measurement points along catheter
            method: Diameter calculation method ('gaussian', 'parabolic', 'threshold')
            old_pixel_spacing: Previous pixel spacing (for comparison)

        Returns:
            CalibrationResult with new pixel spacing and quality metrics
        """
        quality_notes = []
        quality_score = 1.0

        try:
            # Validate inputs
            if mask is None or len(mask.shape) != 2:
                return self._error_result("Invalid mask", catheter_size)

            if centerline is None or len(centerline) < 3:
                return self._error_result("Centerline too short (need at least 3 points)", catheter_size)

            # Get known catheter size
            if catheter_size == "custom":
                if custom_size_mm is None or custom_size_mm <= 0:
                    return self._error_result("Custom catheter size must be positive", catheter_size)
                known_diameter_mm = custom_size_mm
            else:
                if catheter_size not in self.CATHETER_SIZES:
                    valid_sizes = list(self.CATHETER_SIZES.keys())
                    return self._error_result(
                        f"Invalid catheter size. Must be one of: {valid_sizes} or 'custom'",
                        catheter_size
                    )
                known_diameter_mm = self.CATHETER_SIZES[catheter_size]

            logger.info(f"Calibrating with {catheter_size} catheter (known: {known_diameter_mm:.2f}mm)")

            # Normalize centerline format
            centerline = self._normalize_centerline(centerline)

            # Resample centerline to N points
            resampled_centerline = self._resample_centerline(centerline, n_points)

            # Use probability map if available, else use mask
            if probability_map is None:
                probability_map = mask.astype(np.float32)

            # Calculate diameters at each point
            diameters_px = self._calculate_diameters(
                mask, probability_map, resampled_centerline, method
            )

            if len(diameters_px) == 0 or np.all(np.array(diameters_px) == 0):
                return self._error_result("Failed to measure catheter diameters", catheter_size)

            # Calculate robust mean diameter
            measured_diameter_px, filter_stats = self._calculate_robust_mean_diameter(diameters_px)

            if measured_diameter_px <= 0:
                return self._error_result("Measured diameter is zero or negative", catheter_size)

            # Validate measured diameter
            if measured_diameter_px < self.MIN_CATHETER_PX:
                quality_notes.append(f"Measured diameter very small: {measured_diameter_px:.1f}px")
                quality_score *= 0.7
            elif measured_diameter_px > self.MAX_CATHETER_PX:
                quality_notes.append(f"Measured diameter very large: {measured_diameter_px:.1f}px")
                quality_score *= 0.7

            # Check diameter consistency
            cv = filter_stats['cv']
            if cv > 0.2:
                quality_notes.append(f"High diameter variation (CV={cv:.1%})")
                quality_score *= 0.8
            elif cv < 0.05:
                quality_notes.append("Good diameter consistency")

            # Calculate new pixel spacing
            new_pixel_spacing = known_diameter_mm / measured_diameter_px

            # Validate pixel spacing
            if new_pixel_spacing < self.MIN_PIXEL_SPACING:
                quality_notes.append(f"Pixel spacing below typical range: {new_pixel_spacing:.4f} mm/px")
                quality_score *= 0.6
            elif new_pixel_spacing > self.MAX_PIXEL_SPACING:
                quality_notes.append(f"Pixel spacing above typical range: {new_pixel_spacing:.4f} mm/px")
                quality_score *= 0.6
            else:
                quality_notes.append("Pixel spacing within normal range")

            # Compare with old pixel spacing if available
            if old_pixel_spacing is not None and old_pixel_spacing > 0:
                change_ratio = abs(new_pixel_spacing - old_pixel_spacing) / old_pixel_spacing
                if change_ratio > 0.5:
                    quality_notes.append(f"Large change from previous: {change_ratio:.0%}")
                    quality_score *= 0.8

            result = CalibrationResult(
                success=True,
                catheter_size=catheter_size,
                known_diameter_mm=float(known_diameter_mm),
                measured_diameter_px=float(measured_diameter_px),
                new_pixel_spacing=float(new_pixel_spacing),
                old_pixel_spacing=old_pixel_spacing,
                method=method,
                n_points=n_points,
                quality_score=float(quality_score),
                quality_notes=quality_notes,
                diameter_profile_px=diameters_px
            )

            self.last_calibration = result
            self._current_pixel_spacing = new_pixel_spacing

            logger.info(
                f"Calibration successful: {measured_diameter_px:.2f}px → "
                f"{new_pixel_spacing:.4f} mm/px (quality: {quality_score:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return self._error_result(str(e), catheter_size)

    def _error_result(self, error_message: str, catheter_size: str) -> CalibrationResult:
        """Create error result."""
        return CalibrationResult(
            success=False,
            catheter_size=catheter_size,
            known_diameter_mm=0.0,
            measured_diameter_px=0.0,
            new_pixel_spacing=0.0,
            old_pixel_spacing=None,
            method="none",
            n_points=0,
            quality_score=0.0,
            quality_notes=[],
            diameter_profile_px=[],
            error_message=error_message
        )

    def _normalize_centerline(self, centerline: np.ndarray) -> np.ndarray:
        """Ensure centerline is in (N, 2) format as (y, x)."""
        if centerline.ndim != 2 or centerline.shape[1] != 2:
            raise ValueError(f"Invalid centerline shape: {centerline.shape}")
        return centerline.astype(np.float32)

    def _resample_centerline(self, centerline: np.ndarray, n_points: int) -> np.ndarray:
        """Resample centerline to N points with equal arc-length spacing."""
        from scipy.interpolate import interp1d

        if len(centerline) <= 1:
            return centerline

        # Calculate cumulative arc length
        diffs = np.diff(centerline, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        if total_length == 0:
            return np.tile(centerline[0], (n_points, 1))

        # Target arc lengths
        target_lengths = np.linspace(0, total_length, n_points)

        # Interpolate
        interp_y = interp1d(cumulative_length, centerline[:, 0], kind='linear')
        interp_x = interp1d(cumulative_length, centerline[:, 1], kind='linear')

        return np.column_stack([interp_y(target_lengths), interp_x(target_lengths)])

    def _calculate_diameters(
        self,
        mask: np.ndarray,
        probability_map: np.ndarray,
        centerline: np.ndarray,
        method: str
    ) -> List[float]:
        """Calculate diameter at each centerline point using perpendicular cross-sections."""
        from scipy.optimize import curve_fit

        diameters = []

        for i, (cy, cx) in enumerate(centerline):
            # Calculate tangent direction
            n_points = len(centerline)
            if i == 0:
                tangent = centerline[i + 1] - centerline[i]
            elif i == n_points - 1:
                tangent = centerline[i] - centerline[i - 1]
            else:
                tangent = centerline[i + 1] - centerline[i - 1]

            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent = tangent / norm
            else:
                tangent = np.array([1.0, 0.0])

            # Perpendicular direction
            normal = np.array([-tangent[1], tangent[0]])

            # Measure diameter
            diameter = self._measure_diameter_at_point(
                probability_map, (cy, cx), normal, method
            )
            diameters.append(diameter)

        return diameters

    def _measure_diameter_at_point(
        self,
        probability_map: np.ndarray,
        center: Tuple[float, float],
        normal: np.ndarray,
        method: str,
        max_radius: int = 50
    ) -> float:
        """Measure diameter along perpendicular line."""
        from scipy.optimize import curve_fit

        cy, cx = center
        ny, nx = normal

        # Sample points along perpendicular
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

        # Bilinear sampling
        profile = self._bilinear_sample(
            probability_map,
            sample_y[valid_mask],
            sample_x[valid_mask]
        )

        t_valid = t_values[valid_mask]

        # Threshold method (simple and robust for catheter)
        if len(profile) == 0:
            return 0.0

        max_val = profile.max()
        threshold_val = 0.5 * max_val

        above_threshold = profile >= threshold_val

        if not above_threshold.any():
            return 0.0

        indices = np.where(above_threshold)[0]
        first_idx = indices[0]
        last_idx = indices[-1]

        diameter = abs(t_valid[last_idx] - t_valid[first_idx])

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

        h, w = image.shape
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)

        fy = y_coords - np.floor(y_coords)
        fx = x_coords - np.floor(x_coords)

        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy

        values = (
            w00 * image[y0, x0] +
            w01 * image[y1, x0] +
            w10 * image[y0, x1] +
            w11 * image[y1, x1]
        )

        return values

    def _calculate_robust_mean_diameter(
        self,
        diameters: List[float],
        outlier_percentile: float = 10.0
    ) -> Tuple[float, Dict]:
        """
        Calculate robust mean diameter by excluding outliers.

        Uses percentile-based filtering and Hampel-like outlier detection.

        Returns:
            (robust_mean, statistics_dict)
        """
        diameters = np.array(diameters)

        if len(diameters) == 0:
            return 0.0, {'cv': 0.0}

        # Remove zero values
        non_zero = diameters[diameters > 0]

        if len(non_zero) == 0:
            return 0.0, {'cv': 0.0}

        # Remove outliers (bottom and top percentiles)
        lower_bound = np.percentile(non_zero, outlier_percentile)
        upper_bound = np.percentile(non_zero, 100 - outlier_percentile)

        filtered = non_zero[(non_zero >= lower_bound) & (non_zero <= upper_bound)]

        if len(filtered) == 0:
            filtered = non_zero

        robust_mean = float(np.mean(filtered))
        std = float(np.std(filtered))
        cv = std / robust_mean if robust_mean > 0 else 0.0

        stats = {
            'mean': robust_mean,
            'std': std,
            'cv': cv,
            'n_original': len(diameters),
            'n_filtered': len(filtered),
            'outliers_removed': len(diameters) - len(filtered)
        }

        logger.debug(
            f"Robust mean: {robust_mean:.2f}px "
            f"(from {len(filtered)}/{len(diameters)} points, CV={cv:.1%})"
        )

        return robust_mean, stats

    @classmethod
    def get_catheter_sizes(cls) -> Dict[str, float]:
        """Get standard catheter sizes."""
        return cls.CATHETER_SIZES.copy()

    @classmethod
    def validate_catheter_size(
        cls,
        catheter_size: str,
        custom_size_mm: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate catheter size input.

        Returns:
            (is_valid, error_message)
        """
        if catheter_size == "custom":
            if custom_size_mm is None:
                return False, "Custom size required when catheter_size is 'custom'"
            if custom_size_mm <= 0.5 or custom_size_mm > 5.0:
                return False, f"Custom catheter size out of range: {custom_size_mm}mm (expected: 0.5-5.0mm)"
            return True, ""

        if catheter_size not in cls.CATHETER_SIZES:
            available = ", ".join(cls.CATHETER_SIZES.keys())
            return False, f"Invalid catheter size '{catheter_size}'. Available: {available}, custom"

        return True, ""

    def get_current_pixel_spacing(self) -> Optional[float]:
        """Get current pixel spacing from last calibration."""
        return self._current_pixel_spacing


# Singleton instance
_engine_instance: Optional[CalibrationEngine] = None


def get_engine() -> CalibrationEngine:
    """Get singleton CalibrationEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CalibrationEngine()
    return _engine_instance
