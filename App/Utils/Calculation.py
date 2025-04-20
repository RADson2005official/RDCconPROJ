import numpy as np
import logging
import cv2
from typing import Dict, Optional, Tuple, List, Union


class Calculation:
    """
    Enhanced calculation class for material volume and mass calculations.
    This class combines the functionality from the original Calculation class
    and key calculations from MaterialProperties.
    """

    # Define comprehensive material densities (kg/m³)
    MATERIAL_DENSITIES = {
        'Sand': {
            'Dry': 1600.0,
            'Wet': 1900.0,
            'Compacted': 2000.0
        },
        'Gravel': {
            'Loose': 1600.0,
            'Compacted': 1900.0,
            'Crushed': 1650.0,
            'Wet': 1800.0,
            'Dry': 1500.0
        },
        'Soil': {
            'Dry': 1300.0,
            'Wet': 1700.0,
            'Compacted': 1900.0
        },
        'Rock': {
            'Crushed': 1600.0,
            'Solid': 2700.0,
            'Wet': 2800.0
        },
        'Concrete': {
            'Regular': 2400.0,
            'Reinforced': 2500.0,
            'Fresh': 2350.0
        },
        'Other': 1800.0  # Default density
    }

    # Define material specific gravity ranges
    MATERIAL_SPECIFIC_GRAVITY_RANGES = {
        "Sand": (2.65, 2.67),
        "Gravel": (2.5, 3.0),
        "Concrete": (2.3, 2.5),
        "Soil": (1.1, 1.3),
        "Unknown": (0, 0)
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.calibrated_densities = {}
        self.measurement_history = []
        self.MAX_HISTORY_SIZE = 10

    def calculate_volume(self, volume_value: float, material_type: str = "Sand",
                         condition: str = None, material_density: Optional[float] = None,
                         known_mass: Optional[float] = None) -> Dict[str, float]:
        """
        Volume and mass calculation with enhanced material property support

        Args:
            volume_value: The volume value in m³
            material_type: Type of material ("Sand", "Gravel", "Concrete", "Soil", etc.)
            condition: Material condition (e.g., 'Dry', 'Wet', 'Compacted')
            material_density: Optional explicit density value in kg/m³
            known_mass: Optional known mass in kg for calibration

        Returns:
            Dict with volume calculation results
        """
        try:
            # Validate the calculated volume
            if volume_value <= 0 or np.isnan(volume_value) or np.isinf(volume_value):
                self.logger.error(f"Invalid volume calculation result: {volume_value}")
                return {"total": 0.0, "mass": 0.0, "material_type": material_type,
                        "condition": condition, "density": 0.0,
                        "error": "Invalid volume value"}

            # Get density using proper logic
            if material_density is not None:
                # Use explicitly provided density
                self.logger.info(f"Using explicitly provided density: {material_density} kg/m³")
            elif known_mass is not None:
                # Calibrate density using the known mass
                material_density = self.calibrate_density(known_mass, volume_value, material_type)
                self.logger.info(f"Using calibrated density: {material_density} kg/m³")
            else:
                # Use enhanced get_material_density method
                material_density = self.get_material_density(material_type, condition)
                self.logger.info(f"Using material density for {material_type} ({condition}): {material_density} kg/m³")

            # Calculate mass in metric tons
            mass = self.calculate_mass(volume_value, material_type, condition, material_density)

            # Store result in history for consistency checking
            result = {
                "total": volume_value,
                "mass": mass,
                "material_type": material_type,
                "condition": condition,
                "density": material_density,
                "timestamp": None,  # Can be filled with actual timestamp if needed
                "error": None
            }

            # Add to history and maintain maximum size
            self.measurement_history.append(result)
            if len(self.measurement_history) > self.MAX_HISTORY_SIZE:
                self.measurement_history.pop(0)

            # Only perform consistency check if we have enough history
            if len(self.measurement_history) >= 3:
                result = self._check_measurement_consistency(result)

            self.logger.info(f"Final volume calculation: {volume_value:.2f} m³, {mass:.2f} metric tons")
            return result

        except Exception as e:
            self.logger.error(f"Volume calculation failed: {str(e)}")
            return {"total": 0.0, "mass": 0.0, "material_type": material_type,
                    "condition": condition, "density": 0.0,
                    "error": f"Exception: {str(e)}"}

    def get_material_density(self, material_type: str, condition: str = None) -> float:
        """
        Get the density of a material based on its type and condition.

        Args:
            material_type (str): Type of material (e.g., 'Sand', 'Gravel')
            condition (str, optional): Condition of material (e.g., 'Dry', 'Wet', 'Compacted')

        Returns:
            float: Material density in kg/m³
        """
        try:
            material_type = material_type.strip().capitalize() if material_type else "Sand"

            # If material type not found, return default density
            if material_type not in self.MATERIAL_DENSITIES:
                self.logger.warning(f"Unknown material type: {material_type}, using default density")
                return self.MATERIAL_DENSITIES['Other']

            material_data = self.MATERIAL_DENSITIES[material_type]

            # If no condition specified or invalid condition, return average density
            if not condition or not isinstance(material_data, dict):
                return sum(material_data.values()) / len(material_data)

            condition = condition.strip().capitalize() if condition else None
            if condition in material_data:
                density = material_data[condition]
            else:
                # Return average if condition not found
                self.logger.warning(f"Unknown condition: {condition} for {material_type}, using average density")
                density = sum(material_data.values()) / len(material_data)

            # Calculate specific gravity
            specific_gravity = density / 1000  # Convert kg/m³ to specific gravity (water density = 1000 kg/m³)

            # Validate specific gravity against known ranges
            if material_type in self.MATERIAL_SPECIFIC_GRAVITY_RANGES:
                min_sg, max_sg = self.MATERIAL_SPECIFIC_GRAVITY_RANGES[material_type]
                if not (min_sg <= specific_gravity <= max_sg):
                    self.logger.warning(
                        f"Specific gravity {specific_gravity} for {material_type} is outside expected range ({min_sg}-{max_sg})")

            return density

        except Exception as e:
            self.logger.error(f"Error getting material density: {str(e)}")
            return self.MATERIAL_DENSITIES['Other']

    def calculate_mass(self, volume: float, material_type: str, condition: str = None,
                       density: Optional[float] = None) -> float:
        """
        Calculate mass based on volume and material type.

        Args:
            volume (float): Volume in cubic meters
            material_type (str): Type of material
            condition (str, optional): Condition of material
            density (float, optional): Override density if provided

        Returns:
            float: Mass in metric tons
        """
        if density is None:
            density = self.get_material_density(material_type, condition)
        return volume * density / 1000.0  # Convert to metric tons

    def calibrate_density(self, known_mass: float, measured_volume: float, material_type: str = "Sand") -> float:
        """
        Simple density calibration based on known mass and measured volume

        Args:
            known_mass (float): Known mass in kg.
            measured_volume (float): Measured volume in m³.
            material_type (str): Type of material being calibrated.
        Returns:
            float: Calibrated density in kg/m³.
        """
        if measured_volume <= 0:
            self.logger.error("Cannot calibrate with zero or negative volume")
            return self.get_material_density(material_type)

        # Use the actual measured mass/volume for calibration
        calibrated_density = known_mass / measured_volume

        # Store the calibrated density
        self.calibrated_densities[material_type] = calibrated_density

        self.logger.info(f"Calibration for {material_type}:")
        self.logger.info(f"- Calibrated density: {calibrated_density:.2f} kg/m³")

        return calibrated_density

    def _check_measurement_consistency(self, current_result: Dict[str, float]) -> Dict[str, float]:
        """Check if current measurement is consistent with recent history"""
        # Extract recent volumes and masses
        recent_volumes = [r["total"] for r in self.measurement_history[-3:] if "total" in r]
        recent_masses = [r["mass"] for r in self.measurement_history[-3:] if "mass" in r and r["mass"] is not None]

        if not recent_volumes or not recent_masses:
            return current_result

        # Calculate mean and standard deviation
        mean_volume = np.mean(recent_volumes)
        std_volume = np.std(recent_volumes)
        mean_mass = np.mean(recent_masses)
        std_mass = np.std(recent_masses)

        # Check if current measurement is within reasonable bounds (3 standard deviations)
        if std_volume > 0 and abs(current_result["total"] - mean_volume) > 3 * std_volume:
            self.logger.warning(
                f"Volume measurement ({current_result['total']:.2f}) deviates significantly from recent average ({mean_volume:.2f}), std: {std_volume:.2f}")
            # Apply smoothing
            current_result["total"] = 0.7 * current_result["total"] + 0.3 * mean_volume
            self.logger.info(f"Adjusted volume to {current_result['total']:.2f}")

        if std_mass > 0 and abs(current_result["mass"] - mean_mass) > 3 * std_mass:
            self.logger.warning(
                f"Mass measurement ({current_result['mass']:.2f}) deviates significantly from recent average ({mean_mass:.2f}), std: {std_mass:.2f}")
            # Apply smoothing
            current_result["mass"] = 0.7 * current_result["mass"] + 0.3 * mean_mass
            self.logger.info(f"Adjusted mass to {current_result['mass']:.2f}")

        return current_result

    def validate_material_properties(self, material_type: str, density: Optional[float] = None) -> Dict:
        """
        Validate material properties and provide recommendations.

        Args:
            material_type (str): Type of material
            density (float, optional): User-provided density

        Returns:
            Dict: Validation results and recommendations
        """
        result = {
            'valid': True,
            'message': '',
            'recommended_density': None,
            'density_range': None
        }

        try:
            material_type = material_type.strip().capitalize() if material_type else "Sand"
            if material_type not in self.MATERIAL_DENSITIES:
                result['valid'] = False
                result['message'] = f"Unknown material type: {material_type}"
                return result

            material_data = self.MATERIAL_DENSITIES[material_type]
            densities = list(material_data.values())
            min_density = min(densities)
            max_density = max(densities)
            avg_density = sum(densities) / len(densities)

            result['density_range'] = (min_density, max_density)
            result['recommended_density'] = avg_density

            if density is not None:
                if density < min_density * 0.5 or density > max_density * 1.5:
                    result['valid'] = False
                    result[
                        'message'] = f"Density {density} kg/m³ is outside expected range ({min_density}-{max_density} kg/m³)"
                elif density < min_density or density > max_density:
                    result['message'] = f"Warning: Density {density} kg/m³ is unusual for {material_type}"

        except Exception as e:
            result['valid'] = False
            result['message'] = f"Error validating properties: {str(e)}"

        return result

    def get_material_conditions(self, material_type: str) -> list:
        """
        Get available conditions for a material type.

        Args:
            material_type (str): Type of material

        Returns:
            list: Available conditions for the material
        """
        try:
            material_type = material_type.strip().capitalize() if material_type else "Sand"
            if material_type in self.MATERIAL_DENSITIES:
                return list(self.MATERIAL_DENSITIES[material_type].keys())
            return []
        except Exception:
            return []

    def get_available_materials(self) -> list:
        """
        Get list of available material types.

        Returns:
            list: Available material types
        """
        return list(self.MATERIAL_DENSITIES.keys())

    def adjust_sand_density(self, base_density: float,
                            moisture_content: float = 0.0,
                            compaction_level: str = "loose",
                            sand_type: str = "standard") -> float:
        """
        Adjust sand density based on various factors.

        Args:
            base_density: Base density in kg/m³
            moisture_content: Moisture percentage (0-100)
            compaction_level: "loose", "medium", or "dense"
            sand_type: Type of sand ("fine", "coarse", "standard")

        Returns:
            Adjusted density in kg/m³
        """
        # Moisture adjustment factor
        moisture_factor = 1 + (moisture_content / 100)

        # Compaction adjustment
        compaction_factors = {
            "loose": 0.85,
            "medium": 1.0,
            "dense": 1.15
        }
        compaction_factor = compaction_factors.get(compaction_level, 1.0)

        # Sand type adjustment
        sand_type_factors = {
            "fine": 0.95,
            "standard": 1.0,
            "coarse": 1.05
        }
        type_factor = sand_type_factors.get(sand_type, 1.0)

        # Calculate adjusted density
        adjusted_density = base_density * moisture_factor * compaction_factor * type_factor

        self.logger.info(f"Adjusted sand density: {adjusted_density:.2f} kg/m³ "
                         f"(moisture: {moisture_content}%, compaction: {compaction_level}, type: {sand_type})")

        return adjusted_density

    def calculate_conical_pile_volume(self, area: float, height: float) -> float:
        """
        Calculate volume of a conical pile based on base area and height.
        Conical piles typically have 1/3 the volume of a cylinder with the same base and height.

        Args:
            area: Base area in m²
            height: Height in m

        Returns:
            Volume in m³
        """
        cone_factor = 1 / 3
        volume = area * height * cone_factor
        return volume

    def validate_results(self, calculated_result: Dict[str, float],
                         ground_truth: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy metrics for volume and mass."""
        metrics = {}

        # Validate volume
        if 'total' in calculated_result and 'volume' in ground_truth:
            vol_abs_error = abs(calculated_result['total'] - ground_truth['volume'])
            vol_rel_error = vol_abs_error / ground_truth['volume'] * 100 if ground_truth['volume'] > 0 else float('inf')

            metrics.update({
                'volume_absolute_error': vol_abs_error,
                'volume_relative_error': vol_rel_error,
                'volume_rmse': np.sqrt(vol_abs_error ** 2)
            })

        # Validate mass if density is provided
        if 'mass' in calculated_result and 'density' in ground_truth:
            true_mass = ground_truth['volume'] * ground_truth['density'] / 1000.0
            mass_abs_error = abs(calculated_result['mass'] - true_mass)
            mass_rel_error = mass_abs_error / true_mass * 100 if true_mass > 0 else float('inf')

            metrics.update({
                'mass_absolute_error': mass_abs_error,
                'mass_relative_error': mass_rel_error,
                'mass_rmse': np.sqrt(mass_abs_error ** 2)
            })

        return metrics


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
