"""
ECG Parser for Coronary Angiography DICOM

Extracts ECG waveform data and detects R-peaks from DICOM files.
Supports multiple ECG formats including Siemens curved display.

References:
- DICOM Waveform Module (PS3.3 C.10.9)
- Siemens proprietary curve data format (Group 50xx)

Note: R-peak detection is critical for RWS calculation as it defines
cardiac beat boundaries for Dmax/Dmin measurements.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class SiemensECGFilter:
    """
    Filter for Siemens curved ECG display artifacts.

    Siemens angiography systems display ECG in curved segments,
    creating systematic artifacts at transition points.
    """

    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        # Siemens segment durations (seconds)
        self.segment_durations = [2.0, 2.5, 3.0, 4.0]
        # Detection threshold factor
        self.outlier_factor = 8.0
        # Transition window (ms)
        self.transition_window_ms = 100

    def detect_transitions(self, ecg_data: np.ndarray) -> List[int]:
        """Detect screen transition points in Siemens ECG data."""
        if len(ecg_data) < 100:
            return []

        transitions = []

        # Method 1: Detect sudden jumps
        diff = np.abs(np.diff(ecg_data))
        median_diff = np.median(diff)
        std_diff = np.std(diff)
        jump_threshold = max(
            median_diff * self.outlier_factor,
            median_diff + 3 * std_diff
        )
        jump_indices = np.where(diff > jump_threshold)[0]

        # Method 2: Check periodic segment boundaries
        signal_duration = len(ecg_data) / self.sampling_rate
        for segment_duration in self.segment_durations:
            n_segments = int(signal_duration / segment_duration)
            if n_segments > 1:
                for i in range(1, n_segments):
                    boundary_idx = int(i * segment_duration * self.sampling_rate)
                    if boundary_idx < len(diff):
                        window = slice(max(0, boundary_idx - 10), min(len(diff), boundary_idx + 10))
                        if np.max(diff[window]) > jump_threshold:
                            transitions.append(boundary_idx)

        # Combine and merge nearby
        all_trans = sorted(set(list(jump_indices) + transitions))
        merge_dist = int(0.05 * self.sampling_rate)
        merged = []
        for t in all_trans:
            if not merged or t - merged[-1] > merge_dist:
                merged.append(t)

        return merged

    def filter_signal(self, ecg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter ECG signal to remove transition artifacts.

        Returns:
            (filtered_signal, artifact_mask)
        """
        transitions = self.detect_transitions(ecg_data)
        if not transitions:
            return ecg_data.copy(), np.zeros(len(ecg_data), dtype=bool)

        # Create artifact mask
        mask = np.zeros(len(ecg_data), dtype=bool)
        window_samples = int(self.transition_window_ms * self.sampling_rate / 1000)

        for trans_idx in transitions:
            start = max(0, trans_idx - window_samples // 2)
            end = min(len(ecg_data), trans_idx + window_samples // 2)
            mask[start:end] = True

        # Linear interpolation over masked regions
        filtered = ecg_data.copy()
        affected = np.where(mask)[0]

        if len(affected) > 0:
            regions = np.split(affected, np.where(np.diff(affected) != 1)[0] + 1)
            for region in regions:
                if len(region) == 0:
                    continue
                start, end = region[0], region[-1] + 1
                if start > 0 and end < len(filtered):
                    filtered[start:end] = np.linspace(
                        ecg_data[start - 1], ecg_data[end],
                        end - start, endpoint=False
                    )

        return filtered, mask

    def get_suppression_windows(self, transitions: List[int], length: int) -> List[Tuple[int, int]]:
        """Get windows where R-peak detection should be suppressed."""
        windows = []
        suppression = int(0.05 * self.sampling_rate)  # 50ms
        for t in transitions:
            windows.append((max(0, t - suppression), min(length, t + suppression)))
        return windows


class ECGParser:
    """
    Parser for extracting ECG data from coronary angiography DICOM.

    Supports:
    - WaveformSequence (modern DICOM standard)
    - Siemens curved ECG (legacy 50xx group)
    - Private Siemens tags

    Usage:
        parser = ECGParser()
        if parser.extract_from_dicom(dicom_dataset):
            data = parser.get_display_data()
            r_peaks = parser.r_peaks
    """

    def __init__(self):
        self.ecg_data: Optional[np.ndarray] = None
        self.sampling_rate: float = 1000.0
        self.r_peaks: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self._siemens_filter: Optional[SiemensECGFilter] = None

    def extract_from_dicom(self, dicom_dataset) -> bool:
        """
        Extract ECG data from DICOM dataset.

        Tries extraction methods in order:
        1. WaveformSequence (modern)
        2. Legacy Siemens curve (50xx)
        3. Siemens private tags

        Returns:
            True if extraction successful
        """
        self._reset()

        # Try each method
        if self._extract_waveform_sequence(dicom_dataset):
            self._post_process()
            self._detect_r_peaks()
            return True

        if self._extract_siemens_curve(dicom_dataset):
            self._post_process()
            self._detect_r_peaks()
            return True

        if self._extract_siemens_private(dicom_dataset):
            self._post_process()
            self._detect_r_peaks()
            return True

        return False

    def _reset(self):
        """Reset parser state."""
        self.ecg_data = None
        self.sampling_rate = 1000.0
        self.r_peaks = None
        self.metadata = {}
        self._siemens_filter = None

    def _extract_waveform_sequence(self, ds) -> bool:
        """Extract ECG from modern WaveformSequence."""
        try:
            if not hasattr(ds, "WaveformSequence") or len(ds.WaveformSequence) == 0:
                return False

            waveform = ds.WaveformSequence[0]

            # Sampling frequency
            if hasattr(waveform, "SamplingFrequency"):
                self.sampling_rate = float(waveform.SamplingFrequency)

            # Waveform data
            if hasattr(waveform, "WaveformData"):
                bits = int(getattr(waveform, "WaveformBitsAllocated", 16))
                dtype = np.int16 if bits == 16 else np.int8
                self.ecg_data = np.frombuffer(waveform.WaveformData, dtype=dtype).astype(np.float32)

                self.metadata["source"] = "WaveformSequence"
                logger.info(f"Extracted ECG from WaveformSequence: {len(self.ecg_data)} samples @ {self.sampling_rate} Hz")
                return True

        except Exception as e:
            logger.warning(f"WaveformSequence extraction failed: {e}")

        return False

    def _extract_siemens_curve(self, ds) -> bool:
        """
        Extract ECG from Siemens legacy curve data (group 50xx).

        Siemens stores ECG in curved display format with specific characteristics:
        - 12-bit ADC (0-4095 range)
        - Baseline around 2048
        - Stored as unsigned 16-bit
        """
        try:
            from pydicom.tag import Tag

            for group in range(0x5000, 0x5020, 0x0002):
                curve_data_tag = Tag(group, 0x3000)
                curve_samples_tag = Tag(group, 0x0010)

                if curve_data_tag not in ds:
                    continue

                num_points = ds.get(curve_samples_tag, 0)
                if num_points == 0:
                    continue

                raw_data = ds[curve_data_tag].value

                # Check data representation
                repr_tag = Tag(group, 0x0103)
                dtype = np.int16 if repr_tag in ds and ds[repr_tag].value == 1 else np.uint16
                raw_values = np.frombuffer(raw_data, dtype=dtype)

                # Handle dimensions
                dims_tag = Tag(group, 0x0005)
                dimensions = ds.get(dims_tag, 1)
                if dimensions == 2:
                    self.ecg_data = raw_values[:num_points].astype(np.float32)
                else:
                    self.ecg_data = raw_values.astype(np.float32)

                # Calculate sampling rate from video duration
                fps = float(ds.get("CineRate", 15))
                num_frames = int(ds.get("NumberOfFrames", 1))
                video_duration = num_frames / fps
                self.sampling_rate = len(self.ecg_data) / video_duration

                # Process Siemens-specific data
                self._process_siemens_data()

                self.metadata["source"] = f"SiemensCurve_0x{group:04X}"
                logger.info(f"Extracted ECG from Siemens curve: {len(self.ecg_data)} samples @ {self.sampling_rate:.1f} Hz")
                return True

        except Exception as e:
            logger.warning(f"Siemens curve extraction failed: {e}")

        return False

    def _extract_siemens_private(self, ds) -> bool:
        """Extract ECG from Siemens private tags."""
        try:
            from pydicom.tag import Tag

            siemens_tag = Tag(0x0019, 0x1010)
            if siemens_tag not in ds:
                return False

            raw_data = ds[siemens_tag].value
            if len(raw_data) <= 4:
                return False

            # Skip header
            self.ecg_data = np.frombuffer(raw_data[4:], dtype=np.int16).astype(np.float32)
            self.sampling_rate = 1000.0

            self.metadata["source"] = "SiemensPrivate"
            logger.info(f"Extracted ECG from Siemens private: {len(self.ecg_data)} samples")
            return True

        except Exception as e:
            logger.warning(f"Siemens private extraction failed: {e}")

        return False

    def _process_siemens_data(self):
        """Process Siemens ECG data with artifact filtering."""
        if self.ecg_data is None:
            return

        # Remove DC offset (12-bit ADC center)
        baseline = 2048.0
        self.ecg_data = self.ecg_data - baseline

        # Convert to millivolts (typical: 1mV = ~200 ADC units)
        adc_per_mv = 200.0
        self.ecg_data = self.ecg_data / adc_per_mv

        # Apply Siemens filter for screen transitions
        if self.sampling_rate > 0:
            try:
                self._siemens_filter = SiemensECGFilter(self.sampling_rate)
                filtered, mask = self._siemens_filter.filter_signal(self.ecg_data)

                transitions = self._siemens_filter.detect_transitions(self.ecg_data)
                self.ecg_data = filtered

                self.metadata["siemens_filtered"] = True
                self.metadata["transitions_detected"] = len(transitions)
                logger.info(f"Siemens filter: {len(transitions)} transitions filtered")

            except Exception as e:
                logger.warning(f"Siemens filter failed: {e}")
                self.metadata["siemens_filtered"] = False

    def _post_process(self):
        """Apply bandpass filter to ECG data."""
        if self.ecg_data is None or len(self.ecg_data) < 10:
            return

        try:
            from scipy import signal

            # Bandpass filter: 5-40 Hz (typical ECG range)
            nyquist = self.sampling_rate / 2
            low = 5 / nyquist
            high = min(40 / nyquist, 0.99)

            if 0 < low < high < 1:
                b, a = signal.butter(2, [low, high], btype="band")
                self.ecg_data = signal.filtfilt(b, a, self.ecg_data)
                logger.info("Applied bandpass filter to ECG")

        except ImportError:
            logger.warning("scipy not available, skipping bandpass filter")
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")

    def _detect_r_peaks(self) -> Optional[np.ndarray]:
        """
        Detect R-peaks using squared signal with moving average.

        R-peaks define cardiac beat boundaries, critical for RWS calculation.

        Returns:
            Array of R-peak sample indices
        """
        if self.ecg_data is None or len(self.ecg_data) < 10:
            return None

        try:
            from scipy import signal

            # Square signal to emphasize peaks
            squared = self.ecg_data ** 2

            # Moving average
            window_size = max(1, int(0.15 * self.sampling_rate))
            ma = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

            # Find peaks
            threshold = 0.35 * np.max(ma)
            min_distance = int(0.3 * self.sampling_rate)  # Min 300ms between beats
            peaks, _ = signal.find_peaks(ma, height=threshold, distance=min_distance)

            # Refine to actual ECG maxima
            refined = []
            window = int(0.05 * self.sampling_rate)

            # Get suppression windows for Siemens artifacts
            suppression = []
            if self._siemens_filter:
                transitions = self._siemens_filter.detect_transitions(self.ecg_data)
                suppression = self._siemens_filter.get_suppression_windows(transitions, len(self.ecg_data))

            for peak in peaks:
                # Skip if in suppression window
                in_suppression = any(s <= peak <= e for s, e in suppression)
                if in_suppression:
                    continue

                # Refine to local maximum
                start = max(0, peak - window)
                end = min(len(self.ecg_data), peak + window)
                local_max = np.argmax(self.ecg_data[start:end])
                refined.append(start + local_max)

            self.r_peaks = np.array(refined)
            logger.info(f"Detected {len(self.r_peaks)} R-peaks")
            return self.r_peaks

        except ImportError:
            logger.warning("scipy not available, R-peak detection skipped")
        except Exception as e:
            logger.warning(f"R-peak detection failed: {e}")

        return None

    def get_heart_rate(self) -> Optional[float]:
        """Calculate average heart rate (BPM) from R-peaks."""
        if self.r_peaks is None or len(self.r_peaks) < 2:
            return None

        rr_intervals = np.diff(self.r_peaks) / self.sampling_rate
        return 60.0 / np.mean(rr_intervals)

    def get_beat_boundaries(self) -> List[Tuple[int, int]]:
        """
        Get frame indices for each cardiac beat.

        Returns:
            List of (start_frame, end_frame) tuples for each beat
        """
        if self.r_peaks is None or len(self.r_peaks) < 2:
            return []

        # Convert R-peak sample indices to frame indices
        # This requires knowledge of frame rate and ECG-frame sync
        # For now, return sample-based boundaries
        boundaries = []
        for i in range(len(self.r_peaks) - 1):
            boundaries.append((int(self.r_peaks[i]), int(self.r_peaks[i + 1])))

        return boundaries

    def sample_to_frame(self, sample_idx: int, fps: float, num_ecg_samples: int, num_frames: int) -> int:
        """
        Convert ECG sample index to video frame index.

        Args:
            sample_idx: ECG sample index
            fps: Video frame rate
            num_ecg_samples: Total ECG samples
            num_frames: Total video frames

        Returns:
            Corresponding frame index
        """
        # ECG duration in seconds
        ecg_duration = num_ecg_samples / self.sampling_rate
        # Video duration in seconds
        video_duration = num_frames / fps

        # Assume ECG and video are synchronized and same duration
        time_seconds = sample_idx / self.sampling_rate
        frame_idx = int(time_seconds * fps)

        return min(frame_idx, num_frames - 1)

    def get_display_data(self, max_samples: int = 5000) -> Dict[str, Any]:
        """
        Get ECG data formatted for frontend display.

        Downsamples if necessary for efficient transfer.

        Returns:
            Dictionary with signal, time, r_peaks, etc.
        """
        if self.ecg_data is None:
            return {}

        # Downsample if needed
        if len(self.ecg_data) > max_samples:
            step = len(self.ecg_data) // max_samples
            ecg_display = self.ecg_data[::step]
            time_display = np.arange(len(ecg_display)) * step / self.sampling_rate

            # Scale R-peak indices
            r_peak_indices = None
            if self.r_peaks is not None:
                r_peak_indices = [int(p // step) for p in self.r_peaks if p // step < len(ecg_display)]
        else:
            ecg_display = self.ecg_data
            time_display = np.arange(len(ecg_display)) / self.sampling_rate
            r_peak_indices = self.r_peaks.tolist() if self.r_peaks is not None else None

        return {
            "signal": ecg_display.tolist(),
            "time": time_display.tolist(),
            "r_peaks": r_peak_indices,
            "sampling_rate": self.sampling_rate,
            "heart_rate": self.get_heart_rate(),
            "duration": len(self.ecg_data) / self.sampling_rate,
            "metadata": {k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))}
        }


# Singleton instance
_parser_instance: Optional[ECGParser] = None


def get_parser() -> ECGParser:
    """Get singleton ECGParser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = ECGParser()
    return _parser_instance
