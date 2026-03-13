"""Utility nodes -- CLIPSetLastLayer, string ops, logic, math, misc."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Existing: CLIPSetLastLayer
# ---------------------------------------------------------------------------

@registry.register(
    "CLIPSetLastLayer",
    return_types=("CLIP",),
    category="conditioning",
    input_types={"required": {"clip": ("CLIP",), "stop_at_clip_layer": ("INT",)}},
)
def clip_set_last_layer(clip, stop_at_clip_layer):
    if hasattr(clip, "set_last_layer"):
        clip.set_last_layer(stop_at_clip_layer)
    return (clip,)


# ---------------------------------------------------------------------------
# String operations
# ---------------------------------------------------------------------------

@registry.register(
    "StringConcat",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "string1": ("STRING",),
        "string2": ("STRING",),
    }},
)
def string_concat(string1, string2):
    return (string1 + string2,)


@registry.register(
    "StringReplace",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "text": ("STRING",),
        "old": ("STRING",),
        "new": ("STRING",),
    }},
)
def string_replace(text, old, new):
    return (text.replace(old, new),)


@registry.register(
    "StringToInt",
    return_types=("INT",),
    category="utils/string",
    input_types={"required": {"string": ("STRING",)}},
)
def string_to_int(string):
    return (int(string.strip()),)


@registry.register(
    "StringToFloat",
    return_types=("FLOAT",),
    category="utils/string",
    input_types={"required": {"string": ("STRING",)}},
)
def string_to_float(string):
    return (float(string.strip()),)


@registry.register(
    "IntToString",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {"value": ("INT",)}},
)
def int_to_string(value):
    return (str(value),)


@registry.register(
    "FloatToString",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "value": ("FLOAT",),
        "decimals": ("INT",),
    }},
)
def float_to_string(value, decimals=2):
    return (f"{value:.{decimals}f}",)


@registry.register(
    "StringMultiline",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {"text": ("STRING",)}},
)
def string_multiline(text):
    return (text,)


# ---------------------------------------------------------------------------
# Logic operations
# ---------------------------------------------------------------------------

@registry.register(
    "Switch",
    return_types=("*",),
    category="utils/logic",
    input_types={"required": {
        "condition": ("BOOLEAN",),
        "on_true": ("*",),
        "on_false": ("*",),
    }},
)
def switch(condition, on_true, on_false):
    return (on_true if condition else on_false,)


@registry.register(
    "Compare",
    return_types=("BOOLEAN",),
    category="utils/logic",
    input_types={"required": {
        "a": ("FLOAT",),
        "b": ("FLOAT",),
        "operation": ("STRING",),
    }},
)
def compare(a, b, operation="=="):
    ops = {
        "==": a == b,
        "!=": a != b,
        ">": a > b,
        "<": a < b,
        ">=": a >= b,
        "<=": a <= b,
    }
    return (ops.get(operation, False),)


@registry.register(
    "BooleanOp",
    return_types=("BOOLEAN",),
    category="utils/logic",
    input_types={"required": {
        "a": ("BOOLEAN",),
        "operation": ("STRING",),
    },
    "optional": {
        "b": ("BOOLEAN",),
    }},
)
def boolean_op(a, operation="NOT", b=False):
    ops = {
        "AND": a and b,
        "OR": a or b,
        "NOT": not a,
        "XOR": a ^ b,
        "NAND": not (a and b),
        "NOR": not (a or b),
    }
    return (ops.get(operation, False),)


# ---------------------------------------------------------------------------
# Math operations
# ---------------------------------------------------------------------------

@registry.register(
    "IntMath",
    return_types=("INT",),
    category="utils/math",
    input_types={"required": {
        "a": ("INT",),
        "b": ("INT",),
        "operation": ("STRING",),
    }},
)
def int_math(a, b, operation="+"):
    ops = {
        "+": a + b,
        "-": a - b,
        "*": a * b,
        "/": a // b if b != 0 else 0,
        "%": a % b if b != 0 else 0,
        "**": a ** b,
        "min": min(a, b),
        "max": max(a, b),
    }
    return (ops.get(operation, 0),)


@registry.register(
    "FloatMath",
    return_types=("FLOAT",),
    category="utils/math",
    input_types={"required": {
        "a": ("FLOAT",),
        "b": ("FLOAT",),
        "operation": ("STRING",),
    }},
)
def float_math(a, b, operation="+"):
    import math
    ops = {
        "+": a + b,
        "-": a - b,
        "*": a * b,
        "/": a / b if b != 0 else 0.0,
        "**": a ** b if a >= 0 or b == int(b) else 0.0,
        "min": min(a, b),
        "max": max(a, b),
        "abs_diff": abs(a - b),
    }
    return (ops.get(operation, 0.0),)


@registry.register(
    "MathExpression",
    return_types=("FLOAT",),
    category="utils/math",
    input_types={"required": {
        "expression": ("STRING",),
    },
    "optional": {
        "a": ("FLOAT",),
        "b": ("FLOAT",),
        "c": ("FLOAT",),
    }},
)
def math_expression(expression, a=0.0, b=0.0, c=0.0):
    import math as m
    # Safe evaluation with limited namespace
    allowed = {
        "a": a, "b": b, "c": c,
        "abs": abs, "round": round, "min": min, "max": max,
        "int": int, "float": float,
        "pi": m.pi, "e": m.e,
        "sin": m.sin, "cos": m.cos, "tan": m.tan,
        "sqrt": m.sqrt, "log": m.log, "log2": m.log2, "log10": m.log10,
        "floor": m.floor, "ceil": m.ceil,
        "pow": pow,
    }
    result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
    return (float(result),)


# ---------------------------------------------------------------------------
# Misc utility nodes
# ---------------------------------------------------------------------------

@registry.register(
    "NoteNode",
    return_types=(),
    category="utils",
    is_output=True,
    input_types={"required": {"text": ("STRING",)}},
)
def note_node(text):
    return {}


@registry.register(
    "RerouteNode",
    return_types=("*",),
    category="utils",
    input_types={"required": {"input": ("*",)}},
)
def reroute_node(input):
    return (input,)


@registry.register(
    "PrimitiveNode",
    return_types=("*",),
    category="utils",
    input_types={"required": {"value": ("*",)}},
)
def primitive_node(value):
    return (value,)


@registry.register(
    "PreviewAny",
    return_types=(),
    category="utils",
    is_output=True,
    input_types={"required": {"value": ("*",)}},
)
def preview_any(value):
    return {"ui": {"text": [str(value)]}}


@registry.register(
    "IntToFloat",
    return_types=("FLOAT",),
    category="utils/conversion",
    input_types={"required": {"value": ("INT",)}},
)
def int_to_float(value):
    return (float(value),)


@registry.register(
    "FloatToInt",
    return_types=("INT",),
    category="utils/conversion",
    input_types={"required": {
        "value": ("FLOAT",),
        "mode": ("STRING",),
    }},
)
def float_to_int(value, mode="round"):
    if mode == "floor":
        return (int(value // 1),)
    elif mode == "ceil":
        import math
        return (math.ceil(value),)
    else:
        return (round(value),)


@registry.register(
    "SeedNode",
    return_types=("INT",),
    category="utils",
    input_types={"required": {"seed": ("INT",)}},
)
def seed_node(seed):
    return (seed,)


@registry.register(
    "BatchSizeNode",
    return_types=("INT",),
    category="utils",
    input_types={"required": {
        "image": ("IMAGE",),
    }},
)
def batch_size_node(image):
    return (image.shape[0],)


# ---------------------------------------------------------------------------
# Additional utility nodes (template coverage)
# ---------------------------------------------------------------------------

@registry.register(
    "MarkdownNote",
    return_types=(),
    category="utils",
    is_output=True,
    input_types={"required": {"text": ("STRING",)}},
)
def markdown_note(text):
    return {}


@registry.register(
    "Note",
    return_types=(),
    category="utils",
    is_output=True,
    input_types={"required": {"text": ("STRING",)}},
)
def note(text):
    return {}


@registry.register(
    "Reroute",
    return_types=("*",),
    category="utils",
    input_types={"required": {"input": ("*",)}},
)
def reroute(input):
    return (input,)


@registry.register(
    "PrimitiveInt",
    return_types=("INT",),
    category="utils",
    input_types={"required": {"value": ("INT",)}},
)
def primitive_int(value):
    return (value,)


@registry.register(
    "PrimitiveStringMultiline",
    return_types=("STRING",),
    category="utils",
    input_types={"required": {"text": ("STRING",)}},
)
def primitive_string_multiline(text):
    return (text,)


@registry.register(
    "PrimitiveFloat",
    return_types=("FLOAT",),
    category="utils",
    input_types={"required": {"value": ("FLOAT",)}},
)
def primitive_float(value):
    return (value,)


@registry.register(
    "PrimitiveBoolean",
    return_types=("BOOLEAN",),
    category="utils",
    input_types={"required": {"value": ("BOOLEAN",)}},
)
def primitive_boolean(value):
    return (value,)


@registry.register(
    "INTConstant",
    return_types=("INT",),
    category="utils",
    input_types={"required": {"value": ("INT",)}},
)
def int_constant(value):
    return (value,)


@registry.register(
    "RegexReplace",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "text": ("STRING",),
        "pattern": ("STRING",),
        "replacement": ("STRING",),
    }},
)
def regex_replace(text, pattern, replacement):
    import re
    return (re.sub(pattern, replacement, text),)


@registry.register(
    "StringConcatenate",
    return_types=("STRING",),
    category="utils/string",
    input_types={"required": {
        "string1": ("STRING",),
        "string2": ("STRING",),
    }},
)
def string_concatenate(string1, string2):
    return (string1 + string2,)


@registry.register(
    "GetImageSize",
    return_types=("INT", "INT"),
    return_names=["width", "height"],
    category="image",
    input_types={"required": {"image": ("IMAGE",)}},
)
def get_image_size(image):
    return (image.shape[2], image.shape[1])


@registry.register(
    "ImageCompare",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {
        "image1": ("IMAGE",),
        "image2": ("IMAGE",),
    }},
)
def image_compare(image1, image2):
    return {}


@registry.register(
    "SaveGLB",
    return_types=(),
    category="3d",
    is_output=True,
    input_types={"required": {
        "mesh": ("MESH",),
        "filename_prefix": ("STRING",),
    }},
)
def save_glb(mesh, filename_prefix):
    raise NotImplementedError("SaveGLB is not yet implemented")


@registry.register(
    "Preview3D",
    return_types=(),
    category="3d",
    is_output=True,
    input_types={"required": {"mesh": ("MESH",)}},
)
def preview_3d(mesh):
    raise NotImplementedError("Preview3D is not yet implemented")


@registry.register(
    "MaskPreview",
    return_types=(),
    category="mask",
    is_output=True,
    input_types={"required": {"mask": ("MASK",)}},
)
def mask_preview(mask):
    return {}


@registry.register(
    "SaveSVGNode",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {
        "svg_data": ("STRING",),
        "filename_prefix": ("STRING",),
    }},
)
def save_svg_node(svg_data, filename_prefix):
    raise NotImplementedError("SaveSVGNode is not yet implemented")
