"""
Vessel Centerline Extraction

Extracts sub-pixel accurate centerlines from segmentation masks.
Critical for QCA diameter profiling - centerline defines measurement axis.

Methods:
1. Skeleton-based (fast ~50ms): Morphological skeletonization + ordering
2. Distance Transform (sub-pixel): EDT ridge detection
3. Minimum Cost Path (guided): Dijkstra through seed points

References:
- Zhang-Suen thinning algorithm
- Distance Transform: Borgefors, CVGIP 1986
- Skeleton ordering: Nearest-neighbor traversal
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CenterlineExtractor:
    """
    Extract vessel centerline from segmentation mask.

    Centerline extraction is essential for:
    - QCA diameter measurement perpendicular to vessel axis
    - Seed point generation for tracking propagation
    - Vessel length calculation

    Usage:
        extractor = CenterlineExtractor()
        centerline = extractor.extract(mask, method="skeleton")
        sampled, diameters = extractor.get_diameter_profile(mask, centerline, n_points=50)
    """

    def __init__(self):
        """Initialize centerline extractor."""
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for centerline extraction")
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for centerline extraction")

        self.last_centerline: Optional[np.ndarray] = None
        self.last_diameter_map: Optional[np.ndarray] = None

    def extract(
        self,
        mask: np.ndarray,
        method: str = "skeleton",
        seed_points: Optional[List[Tuple[float, float]]] = None,
        probability_map: Optional[np.ndarray] = None,
        num_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract centerline from segmentation mask.

        Args:
            mask: Binary segmentation mask (any dtype, non-zero = vessel)
            method: Extraction method ("skeleton", "distance", "mcp")
            seed_points: Optional (x, y) points for guided extraction
            probability_map: Optional probability map for better accuracy
            num_points: Number of points to sample (for "distance" method)

        Returns:
            Centerline points as (N, 2) array of (y, x) coordinates.
            NOTE: Returns (y, x) format. Swap to (x, y) for canvas display.
        """
        if method == "skeleton":
            return self.extract_skeleton_based(mask, seed_points, probability_map)
        elif method == "distance":
            points, _ = self.extract_distance_transform(mask, num_points or 50)
            return points
        elif method == "mcp":
            if seed_points is None or len(seed_points) < 2:
                raise ValueError("MCP method requires at least 2 seed points")
            return self.extract_minimum_cost_path(mask, seed_points)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'skeleton', 'distance', or 'mcp'")

    def extract_skeleton_based(
        self,
        mask: np.ndarray,
        seed_points: Optional[List[Tuple[float, float]]] = None,
        probability_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract centerline using morphological skeletonization.

        Fast method (~50ms) suitable for real-time tracking.
        Uses Zhang-Suen thinning to reduce vessel to single-pixel width.

        Args:
            mask: Binary segmentation mask
            seed_points: Optional (x, y) seed points to guide endpoint selection
            probability_map: Optional probability map to filter low-confidence regions

        Returns:
            Ordered centerline points as (N, 2) array of (y, x) coordinates
        """
        # Normalize mask to binary
        binary_mask = (mask > 0).astype(np.uint8)

        if binary_mask.sum() == 0:
            logger.warning("Empty segmentation mask")
            return np.array([])

        # Filter by probability if provided
        if probability_map is not None:
            high_prob = (probability_map > 0.3).astype(np.uint8)
            filtered_mask = binary_mask * high_prob
            if filtered_mask.sum() > 0:
                binary_mask = filtered_mask

        # Skeletonize
        skeleton = skeletonize(binary_mask).astype(np.uint8)

        if skeleton.sum() == 0:
            logger.warning("Skeletonization produced empty result")
            return np.array([])

        # Find endpoints
        endpoints = self._find_skeleton_endpoints(skeleton)

        # Use seed points for endpoint selection if available
        if len(endpoints) < 2 and seed_points and len(seed_points) >= 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            # Convert seed (x, y) to (y, x) for comparison
            start_idx = self._find_nearest_point(skel_coords, (seed_points[0][1], seed_points[0][0]))
            end_idx = self._find_nearest_point(skel_coords, (seed_points[-1][1], seed_points[-1][0]))
            endpoints = [tuple(skel_coords[start_idx]), tuple(skel_coords[end_idx])]

        # Fallback: use first/last skeleton points
        if len(endpoints) < 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            if len(skel_coords) > 0:
                endpoints = [tuple(skel_coords[0]), tuple(skel_coords[-1])]
            else:
                return np.array([])

        # Order skeleton points from start to end
        ordered_points = self._order_skeleton_points(skeleton, endpoints[0])

        self.last_centerline = ordered_points
        return ordered_points

    def extract_distance_transform(
        self,
        mask: np.ndarray,
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract centerline using distance transform for sub-pixel accuracy.

        Uses Euclidean Distance Transform (EDT) to find medial axis.
        More accurate but slower than skeleton method.

        Args:
            mask: Binary segmentation mask
            num_points: Number of centerline points to sample

        Returns:
            Tuple of:
            - centerline_points: (N, 2) array of (y, x) coordinates
            - diameter_values: Diameter at each point (pixels)
        """
        binary_mask = (mask > 0).astype(np.uint8)

        if binary_mask.sum() == 0:
            return np.array([]), np.array([])

        # Compute EDT
        edt = ndimage.distance_transform_edt(binary_mask)
        self.last_diameter_map = edt * 2  # Diameter = 2 * radius

        # Skeletonize for point ordering
        skeleton = skeletonize(binary_mask).astype(np.uint8)

        if skeleton.sum() == 0:
            return np.array([]), np.array([])

        # Get ordered points
        endpoints = self._find_skeleton_endpoints(skeleton)
        if len(endpoints) < 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            endpoints = [tuple(skel_coords[0]), tuple(skel_coords[-1])]

        ordered_points = self._order_skeleton_points(skeleton, endpoints[0])

        # Sample N points
        if len(ordered_points) > num_points:
            indices = np.linspace(0, len(ordered_points) - 1, num_points, dtype=int)
            sampled_points = ordered_points[indices]
        else:
            sampled_points = ordered_points

        # Get diameter values
        diameters = np.array([
            edt[int(y), int(x)] * 2
            for y, x in sampled_points
        ])

        self.last_centerline = sampled_points
        return sampled_points, diameters

    def extract_minimum_cost_path(
        self,
        mask: np.ndarray,
        seed_points: List[Tuple[float, float]],
        smooth_sigma: float = 1.0
    ) -> np.ndarray:
        """
        Extract centerline using minimum cost path through seed points.

        Uses Dijkstra's algorithm with cost = 1 / distance_transform.
        Guarantees path passes through all seed points in order.

        Args:
            mask: Binary segmentation mask
            seed_points: List of (x, y) seed points to connect
            smooth_sigma: Gaussian smoothing sigma for path

        Returns:
            Centerline points as (N, 2) array of (y, x) coordinates
        """
        if len(seed_points) < 2:
            raise ValueError("At least 2 seed points required")

        binary_mask = (mask > 0).astype(np.uint8)

        if binary_mask.sum() == 0:
            return np.array([])

        # Create cost map (inverted EDT - lower cost at center)
        edt = ndimage.distance_transform_edt(binary_mask)
        cost_map = 1.0 / (edt + 0.1)

        # Convert seed points (x, y) to (row, col) = (y, x)
        seed_coords = [(int(y), int(x)) for x, y in seed_points]

        # Find path through all seed points
        full_path = []

        for i in range(len(seed_coords) - 1):
            start = seed_coords[i]
            end = seed_coords[i + 1]

            try:
                indices, _ = route_through_array(
                    cost_map, start, end, fully_connected=True
                )

                if i == 0:
                    full_path.extend(indices)
                else:
                    full_path.extend(indices[1:])  # Avoid duplicate points

            except Exception as e:
                logger.warning(f"Path finding failed: {e}, using linear interpolation")
                segment = self._linear_interpolate(start, end)
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])

        path_array = np.array(full_path)

        # Smooth path
        if smooth_sigma > 0 and len(path_array) > 3:
            path_array = self._smooth_path(path_array, smooth_sigma)

        self.last_centerline = path_array
        return path_array

    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find endpoints (pixels with exactly 1 neighbor) in skeleton.

        Uses convolution kernel to count 8-connected neighbors.
        """
        if not CV2_AVAILABLE:
            # Fallback without cv2
            return self._find_endpoints_numpy(skeleton)

        try:
            # Kernel: count neighbors (center = 10 for identification)
            kernel = np.array([[1, 1, 1],
                              [1, 10, 1],
                              [1, 1, 1]], dtype=np.uint8)

            filtered = cv2.filter2D(skeleton, -1, kernel)

            # Endpoints: value = 11 (10 self + 1 neighbor)
            endpoint_mask = (filtered == 11)
            endpoint_coords = np.column_stack(np.where(endpoint_mask))

            return [tuple(coord) for coord in endpoint_coords]
        except AttributeError:
            # Fallback if cv2.filter2D not available
            logger.warning("cv2.filter2D not available, using numpy fallback")
            return self._find_endpoints_numpy(skeleton)

    def _find_endpoints_numpy(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback endpoint finding without OpenCV."""
        from scipy.ndimage import convolve

        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
        endpoints = np.column_stack(np.where((skeleton > 0) & (neighbor_count == 1)))

        return [tuple(coord) for coord in endpoints]

    def _order_skeleton_points(
        self,
        skeleton: np.ndarray,
        start_point: Tuple[int, int]
    ) -> np.ndarray:
        """
        Order skeleton points from start to end using nearest neighbor traversal.

        This ensures centerline points are sequential for proper QCA profiling.
        """
        skel_coords = np.column_stack(np.where(skeleton > 0))
        skel_points = [tuple(coord) for coord in skel_coords]

        if not skel_points:
            return np.array([])

        # Nearest neighbor traversal
        ordered = [start_point]
        remaining = [p for p in skel_points if p != start_point]
        current = start_point

        while remaining:
            # Manhattan distance for speed
            distances = [abs(p[0] - current[0]) + abs(p[1] - current[1]) for p in remaining]
            nearest_idx = np.argmin(distances)
            nearest = remaining[nearest_idx]

            ordered.append(nearest)
            current = nearest
            remaining.pop(nearest_idx)

        return np.array(ordered)

    def _find_nearest_point(self, points: np.ndarray, target: Tuple[float, float]) -> int:
        """Find index of point nearest to target."""
        distances = np.sum((points - np.array(target)) ** 2, axis=1)
        return int(np.argmin(distances))

    def _linear_interpolate(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Linear interpolation between two points."""
        y1, x1 = start
        y2, x2 = end

        distance = max(abs(y2 - y1), abs(x2 - x1))
        num_points = max(2, int(distance))

        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            y = int(y1 + t * (y2 - y1))
            x = int(x1 + t * (x2 - x1))
            points.append((y, x))

        return points

    def _smooth_path(self, path: np.ndarray, sigma: float) -> np.ndarray:
        """Smooth path using Gaussian filter."""
        if len(path) < 3:
            return path

        smoothed_y = ndimage.gaussian_filter1d(path[:, 0].astype(float), sigma=sigma)
        smoothed_x = ndimage.gaussian_filter1d(path[:, 1].astype(float), sigma=sigma)

        return np.column_stack([smoothed_y, smoothed_x])

    def smooth_bspline(
        self,
        points: np.ndarray,
        smoothing: float = 0.0,
        num_output_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Smooth centerline using B-spline interpolation.

        Provides sub-pixel smooth coordinates for accurate diameter profiling.

        Args:
            points: Input points (N, 2) as (y, x)
            smoothing: Smoothing factor (0 = interpolation, >0 = smoothing)
            num_output_points: Number of output points (default: same as input)

        Returns:
            Smoothed points as (M, 2) array
        """
        if len(points) < 4:
            return points

        try:
            # Fit B-spline
            tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing, k=3)

            # Evaluate at desired points
            if num_output_points is None:
                num_output_points = len(points)

            u_new = np.linspace(0, 1, num_output_points)
            smoothed = np.column_stack(splev(u_new, tck))

            return smoothed

        except Exception as e:
            logger.warning(f"B-spline smoothing failed: {e}")
            return points

    def get_diameter_profile(
        self,
        mask: np.ndarray,
        centerline: Optional[np.ndarray] = None,
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get diameter profile along centerline.

        Critical for QCA: returns diameter at each centerline point.

        Args:
            mask: Binary segmentation mask
            centerline: Centerline points (y, x). Uses last extracted if None.
            num_points: Number of points to sample

        Returns:
            Tuple of:
            - sampled_centerline: (N, 2) array of (y, x) coordinates
            - diameters: Diameter at each point (pixels)
        """
        if centerline is None:
            centerline = self.last_centerline

        if centerline is None or len(centerline) == 0:
            return np.array([]), np.array([])

        binary_mask = (mask > 0).astype(np.uint8)
        edt = ndimage.distance_transform_edt(binary_mask)

        # Sample centerline
        if len(centerline) > num_points:
            indices = np.linspace(0, len(centerline) - 1, num_points, dtype=int)
            sampled = centerline[indices]
        else:
            sampled = centerline

        # Get diameters (2 * radius from EDT)
        diameters = np.array([
            edt[int(y), int(x)] * 2
            for y, x in sampled
            if 0 <= int(y) < edt.shape[0] and 0 <= int(x) < edt.shape[1]
        ])

        return sampled[:len(diameters)], diameters

    def generate_seed_points(
        self,
        centerline: np.ndarray,
        num_seeds: int = 3
    ) -> List[Tuple[float, float]]:
        """
        Generate seed points from centerline for tracking propagation.

        Evenly spaced points along centerline for next frame segmentation.

        Args:
            centerline: Centerline points (y, x)
            num_seeds: Number of seed points to generate

        Returns:
            List of (x, y) seed points (NOTE: returns x, y for frontend)
        """
        if len(centerline) < num_seeds:
            # Return all points as seeds
            return [(float(x), float(y)) for y, x in centerline]

        indices = np.linspace(0, len(centerline) - 1, num_seeds, dtype=int)
        seeds = [(float(centerline[i][1]), float(centerline[i][0])) for i in indices]

        return seeds


# Singleton instance
_extractor_instance: Optional[CenterlineExtractor] = None


def get_extractor() -> CenterlineExtractor:
    """Get singleton CenterlineExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = CenterlineExtractor()
    return _extractor_instance
