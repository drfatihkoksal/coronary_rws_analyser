"""
Radial Wall Strain (RWS) Engine

PRIMARY FEATURE: Calculates Radial Wall Strain from diameter measurements
across cardiac cycles with robust outlier detection.

Formula:
    RWS = (Dmax - Dmin) / Dmax × 100%

Where:
    Dmax = Maximum diameter during cardiac cycle
    Dmin = Minimum diameter during cardiac cycle

Clinical Interpretation (Hong et al., EuroIntervention 2023):
    - RWS < 8%: Normal vessel (no significant plaque)
    - RWS 8-12%: Intermediate (possible mild plaque)
    - RWS 12-14%: Vulnerable plaque
    - RWS > 14%: High-risk vulnerable plaque

Measurement Positions:
    1. MLD RWS - Most clinically significant
    2. Proximal Reference RWS
    3. Distal Reference RWS

Features:
    - Hampel filter for robust outlier detection
    - Stenosis-aware processing
    - Quality scoring per measurement
    - Frame-to-frame consistency checks

References:
- Hong et al., "Radial Wall Strain: A Novel Index for
  Coronary Artery Disease", EuroIntervention 2023
- Falk et al., "Coronary Plaque Vulnerability Assessment",
  European Heart Journal 2013
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Outlier detection methods for RWS calculation"""
    NONE = "none"
    HAMPEL = "hampel"
    DOUBLE_HAMPEL = "double_hampel"
    IQR = "iqr"
    TEMPORAL = "temporal"


class HampelFilter:
    """
    Hampel filter for robust outlier detection in coronary diameter measurements.
    Optimized for stenotic vessels and RWS calculation.

    Features:
    - Physiological constraints (0.2-5.5mm vessel diameter)
    - Stenosis-aware adaptive thresholding
    - Frame-to-frame consistency (max 20% change)
    - MAD-based robust statistics
    """

    # Physiological constraints for coronary arteries
    MIN_DIAMETER = 0.2              # Critical stenosis limit (mm)
    MAX_DIAMETER = 5.5              # Maximum LM diameter (mm)
    MAX_RELATIVE_CHANGE = 0.2       # Max 20% change between frames

    # Stenosis grades based on diameter
    STENOSIS_GRADES = {
        'critical': (0.2, 0.8),     # >90% stenosis
        'severe': (0.8, 1.5),       # 70-90% stenosis
        'moderate': (1.5, 2.0),     # 50-70% stenosis
        'mild': (2.0, 2.5),         # 30-50% stenosis
        'normal': (2.5, 5.5),       # <30% stenosis
    }

    @staticmethod
    def detect_stenosis_grade(diameter: float) -> str:
        """Determine stenosis severity from diameter."""
        if diameter < 0.8:
            return 'critical'
        elif diameter < 1.5:
            return 'severe'
        elif diameter < 2.0:
            return 'moderate'
        elif diameter < 2.5:
            return 'mild'
        else:
            return 'normal'

    @staticmethod
    def filter(
        data: np.ndarray,
        window_size: int = 5,
        n_sigmas: float = 3.0,
        adaptive: bool = True,
        preserve_stenosis: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply Hampel filter with stenosis awareness.

        Args:
            data: Input diameter measurements (mm)
            window_size: Window size for median calculation (odd number)
            n_sigmas: Number of MAD deviations for outlier threshold
            adaptive: Use adaptive threshold based on local variance
            preserve_stenosis: Special handling for stenotic measurements

        Returns:
            (filtered_data, outlier_mask, statistics)
        """
        n = len(data)
        filtered = data.copy()
        outlier_mask = np.zeros(n, dtype=bool)

        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2

        # MAD to sigma conversion factor (for normal distribution)
        k = 1.4826

        # Step 1: Apply hard physiological limits
        hard_outliers = (data < HampelFilter.MIN_DIAMETER) | (data > HampelFilter.MAX_DIAMETER)
        filtered[filtered < HampelFilter.MIN_DIAMETER] = HampelFilter.MIN_DIAMETER
        filtered[filtered > HampelFilter.MAX_DIAMETER] = HampelFilter.MAX_DIAMETER
        outlier_mask |= hard_outliers

        # Step 2: Detect stenosis regions
        stenosis_grades = [HampelFilter.detect_stenosis_grade(d) for d in filtered]
        is_stenotic = np.array([g in ['critical', 'severe'] for g in stenosis_grades])

        # Step 3: Apply Hampel filter
        for i in range(n):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Extract window
            window = filtered[start:end]

            if len(window) < 3:
                continue

            # Calculate median and MAD
            median = np.median(window)
            mad = k * np.median(np.abs(window - median))

            # Prevent zero MAD
            mad = max(mad, 0.01)

            # Adaptive threshold
            if adaptive:
                # Calculate local coefficient of variation
                local_cv = mad / (median + 0.01)

                # Adjust threshold based on local variability
                if preserve_stenosis and is_stenotic[i]:
                    # More tolerant for stenotic regions
                    if median < 1.0:  # Critical/severe stenosis
                        # Use absolute threshold for small diameters
                        threshold = max(0.15, n_sigmas * mad)
                    else:
                        threshold = n_sigmas * mad * 1.2
                else:
                    # Normal regions
                    if local_cv > 0.20:  # High variability
                        threshold_multiplier = n_sigmas * 1.2
                    elif local_cv < 0.05:  # Low variability
                        threshold_multiplier = n_sigmas * 0.8
                    else:
                        threshold_multiplier = n_sigmas

                    threshold = threshold_multiplier * mad
            else:
                threshold = n_sigmas * mad

            # Minimum threshold based on vessel size
            min_threshold = 0.1 * median  # At least 10% of local median
            threshold = max(threshold, min_threshold, 0.1)  # Absolute minimum 0.1mm

            # Check if current point is outlier
            deviation = abs(filtered[i] - median)

            if deviation > threshold:
                # Special check for stenotic regions
                if preserve_stenosis and is_stenotic[i] and 0 < i < n - 1:
                    # Check if neighbors are also stenotic
                    if is_stenotic[i-1] and is_stenotic[i+1]:
                        # Likely real stenosis, check relative change
                        rel_change = deviation / median if median > 0 else float('inf')
                        if rel_change > 0.5:  # >50% change is outlier even in stenosis
                            outlier_mask[i] = True
                            filtered[i] = median
                    else:
                        # Isolated stenosis - likely artifact
                        outlier_mask[i] = True
                        filtered[i] = median
                else:
                    outlier_mask[i] = True
                    filtered[i] = median

        # Step 4: Apply frame-to-frame consistency check (max 20% change)
        for i in range(1, n):
            if filtered[i-1] > 0:
                relative_change = abs(filtered[i] - filtered[i-1]) / filtered[i-1]

                if relative_change > HampelFilter.MAX_RELATIVE_CHANGE:
                    outlier_mask[i] = True
                    # Limit the change
                    max_delta = filtered[i-1] * HampelFilter.MAX_RELATIVE_CHANGE
                    if filtered[i] > filtered[i-1]:
                        filtered[i] = filtered[i-1] + max_delta
                    else:
                        filtered[i] = filtered[i-1] - max_delta

        # Calculate statistics
        stenosis_frames = int(np.sum(is_stenotic))
        critical_frames = sum(1 for g in stenosis_grades if g == 'critical')
        severe_frames = sum(1 for g in stenosis_grades if g == 'severe')

        # Check stenosis stability
        stenosis_stable = True
        if stenosis_frames > 0:
            stenotic_values = filtered[is_stenotic]
            if len(stenotic_values) > 1:
                stenosis_cv = np.std(stenotic_values) / (np.mean(stenotic_values) + 0.01)
                stenosis_stable = stenosis_cv < 0.15  # CV < 15% is stable

        stats = {
            'n_outliers': int(np.sum(outlier_mask)),
            'outlier_percentage': float(np.sum(outlier_mask) / n * 100) if n > 0 else 0.0,
            'hard_limit_violations': int(np.sum(hard_outliers)),
            'mean_diameter': float(np.mean(filtered)),
            'std_diameter': float(np.std(filtered)),
            'min_diameter': float(np.min(filtered)),
            'max_diameter': float(np.max(filtered)),
            'stenosis_frames': stenosis_frames,
            'critical_stenosis_frames': critical_frames,
            'severe_stenosis_frames': severe_frames,
            'stenosis_stable': stenosis_stable,
            'predominant_grade': max(set(stenosis_grades), key=stenosis_grades.count) if stenosis_grades else 'unknown'
        }

        return filtered, outlier_mask, stats


@dataclass
class RWSMeasurement:
    """RWS measurement at a specific position."""
    position: str  # "mld", "proximal", "distal"
    dmax: float  # Maximum diameter (mm)
    dmax_frame: int  # Frame index of Dmax
    dmin: float  # Minimum diameter (mm)
    dmin_frame: int  # Frame index of Dmin
    rws: float  # RWS percentage
    interpretation: str  # Clinical interpretation
    quality: Optional[Dict] = None  # Quality report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "dmax": self.dmax,
            "dmax_frame": self.dmax_frame,
            "dmin": self.dmin,
            "dmin_frame": self.dmin_frame,
            "rws": self.rws,
            "interpretation": self.interpretation,
            "quality": self.quality or {}
        }


@dataclass
class RWSResult:
    """Complete RWS analysis result for a beat or frame range."""
    # Individual measurements
    mld_rws: RWSMeasurement  # Most clinically significant
    proximal_rws: RWSMeasurement
    distal_rws: RWSMeasurement

    # Analysis metadata
    beat_number: Optional[int]  # Cardiac beat number (if R-peak sync)
    start_frame: int
    end_frame: int
    num_frames: int

    # Average RWS (for comparison)
    average_rws: float

    # Overall quality
    overall_quality: Optional[Dict] = None

    # Outlier method used
    outlier_method: str = "hampel"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mld_rws": self.mld_rws.to_dict(),
            "proximal_rws": self.proximal_rws.to_dict(),
            "distal_rws": self.distal_rws.to_dict(),
            "beat_number": self.beat_number,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "num_frames": self.num_frames,
            "average_rws": self.average_rws,
            "overall_quality": self.overall_quality or {},
            "outlier_method": self.outlier_method,
            "interpretation": self._get_clinical_interpretation()
        }

    def _get_clinical_interpretation(self) -> Dict[str, Any]:
        """Get clinical interpretation based on MLD RWS."""
        mld_rws = self.mld_rws.rws
        if mld_rws < 8.0:
            return {
                "risk_level": "normal",
                "description": "Normal vessel wall strain",
                "color_code": "#22c55e"  # green
            }
        elif mld_rws < 12.0:
            return {
                "risk_level": "intermediate",
                "description": "Intermediate plaque vulnerability",
                "color_code": "#eab308"  # yellow
            }
        elif mld_rws < 14.0:
            return {
                "risk_level": "vulnerable",
                "description": "Vulnerable plaque detected",
                "color_code": "#f97316"  # orange
            }
        else:
            return {
                "risk_level": "high_risk",
                "description": "High-risk vulnerable plaque",
                "color_code": "#ef4444"  # red
            }


class RWSEngine:
    """
    Radial Wall Strain calculation engine with robust outlier detection.

    Calculates RWS from frame-by-frame diameter measurements using
    the Hampel filter for outlier rejection and quality assessment.

    Usage:
        engine = RWSEngine(outlier_method=OutlierMethod.HAMPEL)

        # Collect QCA metrics for multiple frames
        frame_metrics = {
            0: qca_engine.calculate(mask_0, centerline_0, px_spacing),
            1: qca_engine.calculate(mask_1, centerline_1, px_spacing),
            ...
        }

        # Calculate RWS
        result = engine.calculate(frame_metrics, beat_frames=[10, 25])
    """

    # Clinical thresholds
    RWS_NORMAL_THRESHOLD = 8.0      # RWS < 8% = normal
    RWS_INTERMEDIATE_THRESHOLD = 12.0  # 8-12% = intermediate
    RWS_VULNERABLE_THRESHOLD = 14.0    # 12-14% = vulnerable
    # >14% = high risk

    def __init__(self, outlier_method: OutlierMethod = OutlierMethod.HAMPEL):
        """
        Initialize RWS engine.

        Args:
            outlier_method: Method for outlier detection (default: HAMPEL)
        """
        self.outlier_method = outlier_method
        self._results: List[RWSResult] = []

    def calculate(
        self,
        frame_metrics: Dict[int, Any],  # Dict[frame_index, QCAMetrics]
        beat_frames: Optional[List[int]] = None,
        beat_number: Optional[int] = None
    ) -> Optional[RWSResult]:
        """
        Calculate RWS from frame-by-frame QCA metrics.

        Args:
            frame_metrics: Dictionary mapping frame index to QCAMetrics
            beat_frames: List of frame indices for this cardiac beat.
                        If None, uses all frames in frame_metrics.
            beat_number: Optional beat number for tracking

        Returns:
            RWSResult with all RWS measurements and quality scores
        """
        if not frame_metrics:
            logger.error("No frame metrics provided")
            return None

        try:
            # Determine frame range
            if beat_frames:
                frames = [f for f in beat_frames if f in frame_metrics]
            else:
                frames = sorted(frame_metrics.keys())

            if len(frames) < 2:
                logger.error(f"Need at least 2 frames for RWS, got {len(frames)}")
                return None

            start_frame = frames[0]
            end_frame = frames[-1]

            # Extract diameter series for each position
            mld_series = []
            proximal_series = []
            distal_series = []

            for frame_idx in frames:
                metrics = frame_metrics[frame_idx]

                # Handle both QCAMetrics objects and dicts
                if hasattr(metrics, 'mld'):
                    mld_series.append((frame_idx, metrics.mld))
                    proximal_series.append((frame_idx, metrics.proximal_rd))
                    distal_series.append((frame_idx, metrics.distal_rd))
                elif isinstance(metrics, dict):
                    mld_series.append((frame_idx, metrics.get('mld', metrics.get('mld_mm', 0))))
                    proximal_series.append((frame_idx, metrics.get('proximal_rd', metrics.get('proximal_rd_mm', 0))))
                    distal_series.append((frame_idx, metrics.get('distal_rd', metrics.get('distal_rd_mm', 0))))
                else:
                    logger.warning(f"Unknown metrics type at frame {frame_idx}")
                    continue

            if len(mld_series) < 2:
                logger.error("Not enough valid measurements")
                return None

            # Calculate RWS for each position with outlier detection
            mld_rws = self._calculate_position_rws_robust(mld_series, "mld")
            proximal_rws = self._calculate_position_rws_robust(proximal_series, "proximal")
            distal_rws = self._calculate_position_rws_robust(distal_series, "distal")

            # Calculate average RWS
            average_rws = float(np.mean([mld_rws.rws, proximal_rws.rws, distal_rws.rws]))

            # Overall quality assessment
            quality_scores = []
            for m in [mld_rws, proximal_rws, distal_rws]:
                if m.quality and 'quality_score' in m.quality:
                    quality_scores.append(m.quality['quality_score'])

            overall_quality = {
                'overall_score': float(np.mean(quality_scores)) if quality_scores else 1.0,
                'mld': mld_rws.quality or {},
                'proximal': proximal_rws.quality or {},
                'distal': distal_rws.quality or {}
            }

            result = RWSResult(
                mld_rws=mld_rws,
                proximal_rws=proximal_rws,
                distal_rws=distal_rws,
                beat_number=beat_number,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=len(frames),
                average_rws=average_rws,
                overall_quality=overall_quality,
                outlier_method=self.outlier_method.value
            )

            self._results.append(result)

            # Log with quality info
            quality_status = "Good" if overall_quality['overall_score'] > 0.7 else "Fair" if overall_quality['overall_score'] > 0.5 else "Poor"
            logger.info(
                f"RWS calculated: MLD={mld_rws.rws:.1f}% ({mld_rws.interpretation}), "
                f"Proximal={proximal_rws.rws:.1f}%, Distal={distal_rws.rws:.1f}% | "
                f"Quality: {quality_status} ({overall_quality['overall_score']:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"RWS calculation failed: {e}")
            return None

    def _calculate_position_rws_robust(
        self,
        diameter_series: List[Tuple[int, float]],
        position: str
    ) -> RWSMeasurement:
        """
        Calculate RWS for a specific position with robust outlier detection.

        Args:
            diameter_series: List of (frame_index, diameter) tuples
            position: Position name ("mld", "proximal", "distal")

        Returns:
            RWSMeasurement for this position
        """
        frame_indices = [f for f, _ in diameter_series]
        diameters = np.array([d for _, d in diameter_series])

        if self.outlier_method == OutlierMethod.NONE:
            return self._calculate_position_rws_simple(diameter_series, position)
        elif self.outlier_method == OutlierMethod.HAMPEL:
            return self._calculate_position_rws_hampel(diameters, frame_indices, position)
        elif self.outlier_method == OutlierMethod.DOUBLE_HAMPEL:
            return self._calculate_position_rws_double_hampel(diameters, frame_indices, position)
        else:
            return self._calculate_position_rws_hampel(diameters, frame_indices, position)

    def _calculate_position_rws_simple(
        self,
        diameter_series: List[Tuple[int, float]],
        position: str
    ) -> RWSMeasurement:
        """Calculate RWS without outlier detection (original method)."""
        dmax = -np.inf
        dmax_frame = -1
        dmin = np.inf
        dmin_frame = -1

        for frame_idx, diameter in diameter_series:
            if diameter > dmax:
                dmax = diameter
                dmax_frame = frame_idx
            if diameter < dmin:
                dmin = diameter
                dmin_frame = frame_idx

        # Calculate RWS
        if dmax > 0:
            rws = ((dmax - dmin) / dmax) * 100
        else:
            rws = 0.0

        interpretation = self._interpret_rws(rws)

        return RWSMeasurement(
            position=position,
            dmax=float(dmax),
            dmax_frame=dmax_frame,
            dmin=float(dmin),
            dmin_frame=dmin_frame,
            rws=float(rws),
            interpretation=interpretation,
            quality=None
        )

    def _calculate_position_rws_hampel(
        self,
        diameters: np.ndarray,
        frame_indices: List[int],
        position: str
    ) -> RWSMeasurement:
        """Calculate RWS using Hampel filter for robust outlier removal."""
        if len(diameters) < 2:
            return RWSMeasurement(
                position=position, dmax=0.0, dmax_frame=-1, dmin=0.0, dmin_frame=-1,
                rws=0.0, interpretation="unknown", quality={}
            )

        # Apply Hampel filter
        filtered, outlier_mask, stats = HampelFilter.filter(
            diameters,
            window_size=5,
            n_sigmas=3.0,
            adaptive=True,
            preserve_stenosis=True
        )

        # Quality assessment
        quality_score = 1.0
        quality_notes = []

        # Check if dealing with stenosis
        if stats['predominant_grade'] in ['critical', 'severe']:
            quality_notes.append(f"Stenotic vessel: {stats['predominant_grade']}")

            if not stats['stenosis_stable']:
                quality_score *= 0.8
                quality_notes.append("Unstable stenosis measurements")

            # Use tighter percentiles for stenotic vessels
            max_percentile = 85
            min_percentile = 15
        else:
            # Normal vessel - standard percentiles
            max_percentile = 95
            min_percentile = 5

        # Check outlier rate
        if stats['outlier_percentage'] > 25:
            quality_score *= 0.7
            quality_notes.append(f"High outlier rate: {stats['outlier_percentage']:.1f}%")

        # Find robust extrema using percentiles
        if len(filtered) >= 5:
            dmax = np.percentile(filtered, max_percentile)
            dmin = np.percentile(filtered, min_percentile)
        else:
            dmax = np.max(filtered)
            dmin = np.min(filtered)

        # Find closest actual values
        dmax_idx = int(np.argmin(np.abs(filtered - dmax)))
        dmin_idx = int(np.argmin(np.abs(filtered - dmin)))

        dmax = float(filtered[dmax_idx])
        dmin = float(filtered[dmin_idx])

        dmax_frame = frame_indices[dmax_idx]
        dmin_frame = frame_indices[dmin_idx]

        # Calculate RWS
        if dmax <= 0 or dmax <= dmin:
            rws = 0.0
            quality_score *= 0.5
            quality_notes.append("Invalid diameter relationship")
        else:
            rws = ((dmax - dmin) / dmax) * 100.0

        # Validate RWS based on vessel type
        if stats['predominant_grade'] in ['critical', 'severe'] and rws > 25:
            quality_notes.append("High RWS for stenotic vessel")
            quality_score *= 0.9
        elif stats['predominant_grade'] == 'normal' and rws > 40:
            quality_notes.append("Very high RWS - check tracking")
            quality_score *= 0.8

        interpretation = self._interpret_rws(rws)

        quality_report = {
            'quality_score': float(quality_score),
            'quality_notes': quality_notes,
            'stenosis_grade': stats['predominant_grade'],
            'outliers_removed': stats['n_outliers'],
            'outlier_percentage': stats['outlier_percentage'],
            'mean_diameter': stats['mean_diameter'],
            'diameter_range': [stats['min_diameter'], stats['max_diameter']],
            'percentiles_used': [min_percentile, max_percentile]
        }

        logger.debug(
            f"{position.upper()} RWS (Hampel): {rws:.2f}% | "
            f"Grade: {stats['predominant_grade']} | "
            f"Dmax={dmax:.2f}mm, Dmin={dmin:.2f}mm | "
            f"Quality: {quality_score:.2f}"
        )

        return RWSMeasurement(
            position=position,
            dmax=dmax,
            dmax_frame=dmax_frame,
            dmin=dmin,
            dmin_frame=dmin_frame,
            rws=float(rws),
            interpretation=interpretation,
            quality=quality_report
        )

    def _calculate_position_rws_double_hampel(
        self,
        diameters: np.ndarray,
        frame_indices: List[int],
        position: str
    ) -> RWSMeasurement:
        """Double-pass Hampel filter for extra robustness."""
        if len(diameters) < 2:
            return RWSMeasurement(
                position=position, dmax=0.0, dmax_frame=-1, dmin=0.0, dmin_frame=-1,
                rws=0.0, interpretation="unknown", quality={}
            )

        # First pass: Aggressive outlier removal
        filtered_pass1, outliers_pass1, stats1 = HampelFilter.filter(
            diameters,
            window_size=7,
            n_sigmas=5.0,
            adaptive=False,
            preserve_stenosis=True
        )

        # Second pass: Fine tuning
        filtered_pass2, outliers_pass2, stats2 = HampelFilter.filter(
            filtered_pass1,
            window_size=5,
            n_sigmas=3.0,
            adaptive=True,
            preserve_stenosis=True
        )

        # Combined outlier count
        total_outliers = int(np.sum(outliers_pass1) + np.sum(outliers_pass2))
        outlier_percentage = total_outliers / len(diameters) * 100

        # Find robust max/min
        if len(filtered_pass2) >= 5:
            dmax = np.percentile(filtered_pass2, 90)
            dmin = np.percentile(filtered_pass2, 10)
        else:
            dmax = np.max(filtered_pass2)
            dmin = np.min(filtered_pass2)

        # Find closest indices
        dmax_idx = int(np.argmin(np.abs(filtered_pass2 - dmax)))
        dmin_idx = int(np.argmin(np.abs(filtered_pass2 - dmin)))

        dmax = float(filtered_pass2[dmax_idx])
        dmin = float(filtered_pass2[dmin_idx])

        dmax_frame = frame_indices[dmax_idx]
        dmin_frame = frame_indices[dmin_idx]

        # Calculate RWS
        if dmax <= 0 or dmax <= dmin:
            rws = 0.0
        else:
            rws = ((dmax - dmin) / dmax) * 100.0

        interpretation = self._interpret_rws(rws)

        quality_report = {
            'quality_score': 1.0 - min(outlier_percentage / 100, 0.5),
            'method': 'double_hampel',
            'pass1_outliers': int(np.sum(outliers_pass1)),
            'pass2_outliers': int(np.sum(outliers_pass2)),
            'total_outliers': total_outliers,
            'outlier_percentage': float(outlier_percentage),
            'stenosis_grade': stats2['predominant_grade']
        }

        return RWSMeasurement(
            position=position,
            dmax=dmax,
            dmax_frame=dmax_frame,
            dmin=dmin,
            dmin_frame=dmin_frame,
            rws=float(rws),
            interpretation=interpretation,
            quality=quality_report
        )

    def _interpret_rws(self, rws: float) -> str:
        """Provide clinical interpretation of RWS value."""
        if rws < self.RWS_NORMAL_THRESHOLD:
            return "normal"
        elif rws < self.RWS_INTERMEDIATE_THRESHOLD:
            return "intermediate"
        elif rws < self.RWS_VULNERABLE_THRESHOLD:
            return "vulnerable"
        else:
            return "high_risk"

    def calculate_from_diameters(
        self,
        mld_diameters: List[float],
        proximal_diameters: List[float],
        distal_diameters: List[float],
        frame_indices: Optional[List[int]] = None,
        beat_number: Optional[int] = None
    ) -> Optional[RWSResult]:
        """
        Calculate RWS directly from diameter arrays.

        Convenience method when QCAMetrics objects aren't available.

        Args:
            mld_diameters: MLD values per frame
            proximal_diameters: Proximal RD values per frame
            distal_diameters: Distal RD values per frame
            frame_indices: Frame indices (default: 0, 1, 2, ...)
            beat_number: Optional beat number

        Returns:
            RWSResult
        """
        if frame_indices is None:
            frame_indices = list(range(len(mld_diameters)))

        # Convert to format expected by calculate()
        frame_metrics = {}
        for i, frame_idx in enumerate(frame_indices):
            if i < len(mld_diameters) and i < len(proximal_diameters) and i < len(distal_diameters):
                frame_metrics[frame_idx] = {
                    'mld': mld_diameters[i],
                    'proximal_rd': proximal_diameters[i],
                    'distal_rd': distal_diameters[i]
                }

        return self.calculate(frame_metrics, beat_number=beat_number)

    def get_all_results(self) -> List[RWSResult]:
        """Get all calculated RWS results."""
        return self._results.copy()

    def get_beat_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all analyzed beats.

        Returns:
            Summary dictionary with mean, std, min, max for each position
        """
        if not self._results:
            return {}

        mld_rws_values = [r.mld_rws.rws for r in self._results]
        proximal_rws_values = [r.proximal_rws.rws for r in self._results]
        distal_rws_values = [r.distal_rws.rws for r in self._results]

        def stats(values: List[float]) -> Dict[str, float]:
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values)
            }

        return {
            "mld_rws": stats(mld_rws_values),
            "proximal_rws": stats(proximal_rws_values),
            "distal_rws": stats(distal_rws_values),
            "num_beats": len(self._results)
        }

    def clear_results(self):
        """Clear all stored results."""
        self._results = []

    def export_for_analysis(self) -> List[Dict[str, Any]]:
        """
        Export results in format suitable for statistical analysis.

        Returns:
            List of dictionaries, one per beat
        """
        export_data = []

        for i, result in enumerate(self._results):
            export_data.append({
                "beat_index": i,
                "beat_number": result.beat_number,
                "start_frame": result.start_frame,
                "end_frame": result.end_frame,
                "num_frames": result.num_frames,
                "outlier_method": result.outlier_method,
                # MLD RWS (most important)
                "mld_rws": result.mld_rws.rws,
                "mld_dmax": result.mld_rws.dmax,
                "mld_dmin": result.mld_rws.dmin,
                "mld_dmax_frame": result.mld_rws.dmax_frame,
                "mld_dmin_frame": result.mld_rws.dmin_frame,
                "mld_interpretation": result.mld_rws.interpretation,
                "mld_quality_score": result.mld_rws.quality.get('quality_score', 1.0) if result.mld_rws.quality else 1.0,
                # Proximal RWS
                "proximal_rws": result.proximal_rws.rws,
                "proximal_dmax": result.proximal_rws.dmax,
                "proximal_dmin": result.proximal_rws.dmin,
                # Distal RWS
                "distal_rws": result.distal_rws.rws,
                "distal_dmax": result.distal_rws.dmax,
                "distal_dmin": result.distal_rws.dmin,
                # Average
                "average_rws": result.average_rws,
                # Overall quality
                "overall_quality_score": result.overall_quality.get('overall_score', 1.0) if result.overall_quality else 1.0
            })

        return export_data


# Singleton instance
_engine_instance: Optional[RWSEngine] = None


def get_engine(outlier_method: OutlierMethod = OutlierMethod.HAMPEL) -> RWSEngine:
    """Get singleton RWSEngine instance."""
    global _engine_instance
    if _engine_instance is None or _engine_instance.outlier_method != outlier_method:
        _engine_instance = RWSEngine(outlier_method=outlier_method)
    return _engine_instance
