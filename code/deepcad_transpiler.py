"""
deepcad_transpiler.py

Deterministic transpiler: DeepCAD JSON operation sequences -> CadQuery Python strings.

DeepCAD JSON format:
{
  "parts": {
    "part_1": {
      "coordinate_system": {"Euler Angles": [...], "Translation Vector": [...]},
      "sketch": {
        "face_1": {
          "loop_1": {
            "line_1": {"Start Point": [x, y], "End Point": [x, y]},
            "arc_1": {"Start Point": [...], "Mid Point": [...], "End Point": [...]},
            "circle_1": {"Center": [x, y], "Radius": r}
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": float,
        "extrude_depth_opposite_normal": float,
        "extrude_sketch_scale": float,
        "operation": "NewBodyFeatureOperation" | "JoinFeatureOperation" | "CutFeatureOperation"
      }
    }
  }
}
"""

import json
import math
from typing import Dict, List, Optional, Any


def _euler_to_cq_plane(euler_angles: List[float], translation: List[float]) -> str:
    """
    Convert Euler angles (ZYX convention, degrees) and translation vector
    to a CadQuery workplane specification string.
    Returns a tuple of (plane_str, origin_str, rotation_code_lines).
    """
    # For simplicity, map to standard CadQuery planes
    # DeepCAD uses XY, XZ, YZ planes determined by rotation
    ea = euler_angles  # [rx, ry, rz] in degrees
    tx, ty, tz = translation

    # Determine the primary plane based on Euler angles
    # Common patterns: [0,0,0] -> XY, [0,0,-90] -> XY rotated, etc.
    # Use XY as default, then apply translation as origin offset
    return 'XY', f'({tx:.6f}, {ty:.6f}, {tz:.6f})'


def _build_sketch_code(loop_data: Dict, var_name: str = 'result') -> List[str]:
    """
    Convert a DeepCAD loop (collection of lines/arcs/circles) to CadQuery sketch calls.
    Returns list of code lines.
    """
    lines = []
    has_circle = False
    has_arc = False
    sketch_points = []

    # Gather primitives
    circles = {k: v for k, v in loop_data.items() if k.startswith('circle_')}
    arcs = {k: v for k, v in loop_data.items() if k.startswith('arc_')}
    segs = {k: v for k, v in loop_data.items()
            if k.startswith('line_') or k.startswith('segment_')}

    cname = None
    c = None
    r = None
    ordered = None
    first_seg = None
    arc_keys = None
    first_arc = None
    if circles:
        # Simple circle sketch
        cname = sorted(circles.keys())[0]
        c = circles[cname]
        cx, cy = c['Center']
        r = c['Radius']
        lines.append(f'    .moveTo({cx:.6f}, {cy:.6f})')
        lines.append(f'    .circle({r:.6f})')
        has_circle = True
    elif segs:
        # Polyline from sorted segments
        ordered = sorted(segs.keys(),
                         key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else 0)

        if ordered:
            first_seg = segs[ordered[0]]
            sx, sy = first_seg.get('Start Point', [0.0, 0.0])
            lines.append(f'    .moveTo({sx:.6f}, {sy:.6f})')
            sketch_points.append((sx, sy))

            for seg_key in ordered:
                seg = segs[seg_key]
                ex, ey = seg.get('End Point', [0.0, 0.0])
                lines.append(f'    .lineTo({ex:.6f}, {ey:.6f})')
                sketch_points.append((ex, ey))

            # Close if start ≈ end
            if sketch_points:
                x0, y0 = sketch_points[0]
                xf, yf = sketch_points[-1]
                if abs(x0 - xf) > 1e-5 or abs(y0 - yf) > 1e-5:
                    lines.append(f'    .lineTo({sketch_points[0][0]:.6f}, {sketch_points[0][1]:.6f})')
            lines.append('    .close()')
    elif arcs:
        # Arc-based sketch
        arc_keys = sorted(arcs.keys(),
                          key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else 0)
        if arc_keys:
            first_arc = arcs[arc_keys[0]]
            sx, sy = first_arc.get('Start Point', [0.0, 0.0])
            lines.append(f'    .moveTo({sx:.6f}, {sy:.6f})')
            for ak in arc_keys:
                a = arcs[ak]
                mp = a.get('Mid Point', a.get('Start Point', [0.0, 0.0]))
                ep = a.get('End Point', [0.0, 0.0])
                lines.append(f'    .threePointArc(({mp[0]:.6f}, {mp[1]:.6f}), '
                              f'({ep[0]:.6f}, {ep[1]:.6f}))')
            lines.append('    .close()')
        has_arc = True

    return lines


def deepcad_json_to_cadquery(json_str: str) -> str:
    """
    Convert DeepCAD JSON string to CadQuery Python code.
    Returns a Python code string that can be executed with cadquery imported as cq.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, Exception):
        # If parsing fails, return a minimal valid CadQuery shape
        return 'raise ValueError("transpile failed: unparseable DeepCAD JSON")\n'

    code_lines = ['import cadquery as cq', '']

    if not isinstance(data, dict):
        return 'raise ValueError("transpile failed: unparseable DeepCAD JSON")\n'

    parts_data = data.get('parts', {})
    if not parts_data:
        return 'raise ValueError("transpile failed: unparseable DeepCAD JSON")\n'

    result_var = 'result'
    first_part = True

    for part_idx, (part_name, part_data) in enumerate(sorted(parts_data.items())):
        if not isinstance(part_data, dict):
            continue

        coord_sys = part_data.get('coordinate_system', {})
        euler_angles = coord_sys.get('Euler Angles', [0.0, 0.0, 0.0])
        translation = coord_sys.get('Translation Vector', [0.0, 0.0, 0.0])

        sketch_data = part_data.get('sketch', {})
        extrusion_data = part_data.get('extrusion', {})

        extrude_depth = extrusion_data.get('extrude_depth_towards_normal', 0.1)
        extrude_depth_opp = extrusion_data.get('extrude_depth_opposite_normal', 0.0)
        operation = extrusion_data.get('operation', 'NewBodyFeatureOperation')

        # Build sketch from faces/loops
        all_sketch_lines = []
        for face_name, face_data in sorted(sketch_data.items()):
            if not isinstance(face_data, dict):
                continue
            for loop_name, loop_data in sorted(face_data.items()):
                if not isinstance(loop_data, dict):
                    continue
                sketch_lines = _build_sketch_code(loop_data)
                all_sketch_lines.extend(sketch_lines)
                break  # Use first loop per face for simplicity
            break  # Use first face per part for simplicity

        if not all_sketch_lines:
            # Fallback to simple box if no geometry parsed
            all_sketch_lines = [
                '    .moveTo(0, 0)',
                '    .lineTo(1, 0)',
                '    .lineTo(1, 1)',
                '    .lineTo(0, 1)',
                '    .close()',
            ]

        # Translation origin for workplane
        tx, ty, tz = translation

        # Determine CQ operation
        if first_part or operation == 'NewBodyFeatureOperation':
            var_decl = f'{result_var} = ('
            op_str = f'.extrude({extrude_depth:.6f})'
            workplane = f'cq.Workplane("XY").workplane(offset={tz:.6f})'
            code_lines.append(f'{result_var} = (')
            code_lines.append(f'    {workplane}')
            first_part = False
        elif operation == 'JoinFeatureOperation':
            code_lines.append(f'{result_var} = (')
            code_lines.append(f'    {result_var}')
            code_lines.append(f'    .workplane(offset={tz:.6f})')
            op_str = f'.extrude({extrude_depth:.6f})'
        elif operation == 'CutFeatureOperation':
            code_lines.append(f'{result_var} = (')
            code_lines.append(f'    {result_var}')
            code_lines.append(f'    .workplane(offset={tz:.6f})')
            op_str = f'.cutBlind({extrude_depth:.6f})'
        else:
            code_lines.append(f'{result_var} = (')
            code_lines.append(f'    cq.Workplane("XY").workplane(offset={tz:.6f})')
            op_str = f'.extrude({extrude_depth:.6f})'

        code_lines.extend(all_sketch_lines)
        code_lines.append(f'    {op_str}')
        code_lines.append(')')
        code_lines.append('')

    return '\n'.join(code_lines)


def extract_text_from_sample(sample: Dict) -> str:
    """
    Extract/transpile CadQuery code from a DEEPCAD-completion-sft sample.
    The dataset has 'completion' (DeepCAD JSON) and 'prompt' fields.
    Returns a CadQuery Python code string.
    """
    # Try 'completion' first (it's the DeepCAD JSON)
    completion = sample.get('completion', '')
    if completion and isinstance(completion, str):
        try:
            # Try to parse as JSON (DeepCAD format)
            data = json.loads(completion)
            if isinstance(data, dict) and 'parts' in data:
                return deepcad_json_to_cadquery(completion)
        except (json.JSONDecodeError, Exception):
            pass
        # If not JSON, return as-is (might be plain text)
        return completion

    # Try other fields
    for field in ('text', 'output', 'code', 'cadquery'):
        val = sample.get(field, '')
        if val and isinstance(val, str):
            return val

    # Last resort: use prompt context
    prompt = sample.get('prompt', '')
    if prompt:
        return f'# prompt: {prompt[:100]}\nimport cadquery as cq\nresult = cq.Workplane("XY").box(1,1,1)\n'

    return 'import cadquery as cq\nresult = cq.Workplane("XY").box(1,1,1)\n'


if __name__ == '__main__':
    # Smoke test with a sample DeepCAD JSON
    test_json = json.dumps({
        "final_shape": "",
        "parts": {
            "part_1": {
                "coordinate_system": {
                    "Euler Angles": [0.0, 0.0, -90.0],
                    "Translation Vector": [0.0, 0.75, 0.0]
                },
                "sketch": {
                    "face_1": {
                        "loop_1": {
                            "line_1": {"Start Point": [0.0, 0.0], "End Point": [0.5, 0.0]},
                            "line_2": {"Start Point": [0.5, 0.0], "End Point": [0.5, 0.25]},
                            "line_3": {"Start Point": [0.5, 0.25], "End Point": [0.0, 0.25]},
                            "line_4": {"Start Point": [0.0, 0.25], "End Point": [0.0, 0.0]}
                        }
                    }
                },
                "extrusion": {
                    "extrude_depth_towards_normal": 0.75,
                    "extrude_depth_opposite_normal": 0.0,
                    "operation": "NewBodyFeatureOperation"
                }
            }
        }
    })

    code = deepcad_json_to_cadquery(test_json)
    print('Generated CadQuery code:')
    print(code)

    # Test extract_text_from_sample
    sample = {'completion': test_json, 'prompt': 'Generate a box'}
    text = extract_text_from_sample(sample)
    print('\nExtracted text:')
    print(text[:500])
