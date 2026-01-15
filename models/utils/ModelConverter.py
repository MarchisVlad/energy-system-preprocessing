"""
Model Format Converter
Supports MPS (Mathematical Programming System), GDX (GAMS Data Exchange), and GMS (GAMS Model) formats
"""

import os
import re
from enum import Enum
from typing import Optional, Dict, List, Tuple


class ModelFormat(Enum):
    """Supported model file formats"""
    MPS = "mps"
    GDX = "gdx"
    GMS = "gms"
    UNKNOWN = "unknown"


class ModelConverter:
    """Convert between different optimization model formats"""
    
    def __init__(self):
        self.format_signatures = {
            ModelFormat.MPS: [b'NAME', b'ROWS', b'COLUMNS', b'RHS'],
            ModelFormat.GDX: [b'GDX', b'\x00\x00\x00'],
            ModelFormat.GMS: [b'$', b'SETS', b'PARAMETERS', b'VARIABLES', b'EQUATIONS']
        }
    
    def detect_format(self, filepath: str) -> ModelFormat:
        """
        Detect the format of a model file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            ModelFormat enum indicating the detected format
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check by extension first
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.mps', '.mps.gz']:
            return ModelFormat.MPS
        elif ext == '.gdx':
            return ModelFormat.GDX
        elif ext == '.gms':
            return ModelFormat.GMS
        
        # Check by content
        try:
            with open(filepath, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                
            # Check for GDX binary signature
            if content[:3] == b'GDX' or b'\x00\x00\x00' in content[:100]:
                return ModelFormat.GDX
            
            # Convert to text for text-based formats
            try:
                text_content = content.decode('utf-8', errors='ignore').upper()
            except:
                return ModelFormat.UNKNOWN
            
            # Check for MPS format
            if any(keyword in text_content for keyword in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'ENDATA']):
                mps_score = sum(1 for kw in ['NAME', 'ROWS', 'COLUMNS', 'RHS'] if kw in text_content)
                if mps_score >= 2:
                    return ModelFormat.MPS
            
            # Check for GMS format
            if any(keyword in text_content for keyword in ['SETS', 'SET', 'PARAMETERS', 'PARAMETER', 'VARIABLES', 'VARIABLE', 'EQUATIONS', 'EQUATION', 'MODEL', 'SOLVE']):
                gms_score = sum(1 for kw in ['SET', 'PARAMETER', 'VARIABLE', 'EQUATION', 'MODEL'] if kw in text_content)
                if gms_score >= 2:
                    return ModelFormat.GMS
            
        except Exception as e:
            print(f"Error detecting format: {e}")
            return ModelFormat.UNKNOWN
        
        return ModelFormat.UNKNOWN
    
    def convert(self, input_path: str, output_path: str, target_format: Optional[ModelFormat] = None) -> bool:
        """
        Convert a model file to another format
        
        Args:
            input_path: Path to input model file
            output_path: Path to output model file
            target_format: Target format (if None, inferred from output_path extension)
            
        Returns:
            True if conversion successful, False otherwise
        """
        # Detect input format
        source_format = self.detect_format(input_path)
        if source_format == ModelFormat.UNKNOWN:
            print(f"Unable to detect format of {input_path}")
            return False
        
        # Determine target format
        if target_format is None:
            ext = os.path.splitext(output_path)[1].lower()
            format_map = {'.mps': ModelFormat.MPS, '.gdx': ModelFormat.GDX, '.gms': ModelFormat.GMS}
            target_format = format_map.get(ext, ModelFormat.UNKNOWN)
        
        if target_format == ModelFormat.UNKNOWN:
            print(f"Unable to determine target format for {output_path}")
            return False
        
        print(f"Converting from {source_format.value.upper()} to {target_format.value.upper()}...")
        
        # Perform conversion
        try:
            if source_format == ModelFormat.MPS and target_format == ModelFormat.GMS:
                return self._mps_to_gms(input_path, output_path)
            elif source_format == ModelFormat.GMS and target_format == ModelFormat.MPS:
                return self._gms_to_mps(input_path, output_path)
            elif source_format == ModelFormat.GDX:
                print("GDX format requires GAMS tools for conversion")
                print("Please use gdxdump or gdx2sqlite utilities")
                return False
            elif target_format == ModelFormat.GDX:
                print("GDX format requires GAMS tools for conversion")
                print("Please use gdxxrw or other GAMS utilities")
                return False
            else:
                print(f"Conversion from {source_format.value} to {target_format.value} not yet implemented")
                return False
        except Exception as e:
            print(f"Conversion error: {e}")
            return False
    
    def _mps_to_gms(self, input_path: str, output_path: str) -> bool:
        """Convert MPS format to GMS format"""
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        gms_lines = ["* GAMS model converted from MPS format\n\n"]
        
        # Parse MPS sections
        section = None
        rows = {}
        columns = set()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            if line.startswith('NAME'):
                model_name = line.split()[1] if len(line.split()) > 1 else "MODEL"
                gms_lines.append(f"* Model: {model_name}\n\n")
            elif line.startswith('ROWS'):
                section = 'ROWS'
            elif line.startswith('COLUMNS'):
                section = 'COLUMNS'
            elif line.startswith('RHS'):
                section = 'RHS'
            elif line.startswith('BOUNDS'):
                section = 'BOUNDS'
            elif line.startswith('ENDATA'):
                break
            elif section == 'ROWS':
                parts = line.split()
                if len(parts) >= 2:
                    rows[parts[1]] = parts[0]
        
        # Generate GAMS model structure
        gms_lines.append("VARIABLES\n")
        gms_lines.append("    obj      objective function\n")
        gms_lines.append(";\n\n")
        
        gms_lines.append("EQUATIONS\n")
        for row_name in rows:
            gms_lines.append(f"    {row_name}\n")
        gms_lines.append(";\n\n")
        
        gms_lines.append("* Model equations would be defined here\n")
        gms_lines.append("* (Full MPS to GMS conversion requires parsing coefficients)\n\n")
        
        gms_lines.append("MODEL mpsmodel /all/;\n")
        gms_lines.append("SOLVE mpsmodel USING LP MINIMIZING obj;\n")
        
        with open(output_path, 'w') as f:
            f.writelines(gms_lines)
        
        return True
    
    def _gms_to_mps(self, input_path: str, output_path: str) -> bool:
        """Convert GMS format to MPS format (simplified)"""
        with open(input_path, 'r') as f:
            content = f.read()
        
        mps_lines = ["NAME          GMSMODEL\n"]
        mps_lines.append("ROWS\n")
        mps_lines.append(" N  OBJ\n")
        
        # Extract equation names from EQUATIONS section
        equations_match = re.search(r'EQUATIONS\s+(.*?);', content, re.DOTALL | re.IGNORECASE)
        if equations_match:
            eq_text = equations_match.group(1)
            equations = [e.strip() for e in eq_text.split('\n') if e.strip() and not e.strip().startswith('*')]
            for eq in equations:
                eq_name = eq.split()[0] if eq.split() else eq
                mps_lines.append(f" E  {eq_name}\n")
        
        mps_lines.append("COLUMNS\n")
        
        # Extract variable names
        variables_match = re.search(r'VARIABLES\s+(.*?);', content, re.DOTALL | re.IGNORECASE)
        if variables_match:
            var_text = variables_match.group(1)
            variables = [v.strip() for v in var_text.split('\n') if v.strip() and not v.strip().startswith('*')]
            for var in variables[:5]:  # Limit for example
                var_name = var.split()[0] if var.split() else var
                mps_lines.append(f"    {var_name:10} OBJ       1.0\n")
        
        mps_lines.append("RHS\n")
        mps_lines.append("BOUNDS\n")
        mps_lines.append("ENDATA\n")
        
        with open(output_path, 'w') as f:
            f.writelines(mps_lines)
        
        return True


# Example usage
if __name__ == "__main__":
    converter = ModelConverter()
    
    # Example: detect format
    test_file = "model.mps"  # Replace with your file
    if os.path.exists(test_file):
        fmt = converter.detect_format(test_file)
        print(f"Detected format: {fmt.value}")
        
        # Example: convert
        # converter.convert("model.mps", "model.gms")
    else:
        print("Example file not found. Create a model file to test.")
        print("\nUsage:")
        print("  converter = ModelConverter()")
        print("  converter.detect_format('model.mps')")
        print("  converter.convert('model.mps', 'model.gms')")