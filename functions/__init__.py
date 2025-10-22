"""
Paquete de funciones auxiliares para detección de manzanas
"""

from .analysys import (
    analyze_apple_quality,
    analyze_apple_quality_custom,
    analyze_apple_quality_with_logging,
    analyze_color_distribution,
    calculate_color_confidence,
    classify_apple_by_color,
    classify_apple_with_custom_ranges,
    configure_apple_color_ranges,
    detect_wrinkled_texture,
    get_apple_quality_details,
    get_dominant_rgb_color,
)

__all__ = [
    'analyze_apple_quality',
    'analyze_apple_quality_with_logging',
    'get_apple_quality_details',
    'configure_apple_color_ranges',
    'analyze_apple_quality_custom',
    'get_dominant_rgb_color',
    'classify_apple_by_color',
    'calculate_color_confidence',
    'analyze_color_distribution',
    'classify_apple_with_custom_ranges',
    'detect_wrinkled_texture'
]
