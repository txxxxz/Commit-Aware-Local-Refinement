"""Completion cleanup and syntax checks used by evaluation."""

from ._runtime import (
    _extract_body_from_full_function,
    _extract_code_completion,
    _extract_humaneval_body_completion,
    _is_postprocess_stop_line,
    _line_indent_width,
    _normalize_function_body_lines,
    _obvious_truncation_for_eval,
    _strip_markdown_fence_region,
    _target_function_present_for_eval,
    format_success,
    parse_success,
    parse_success_for_eval,
    postprocess_completion_for_eval,
)

__all__ = [
    "_extract_body_from_full_function",
    "_extract_code_completion",
    "_extract_humaneval_body_completion",
    "_is_postprocess_stop_line",
    "_line_indent_width",
    "_normalize_function_body_lines",
    "_obvious_truncation_for_eval",
    "_strip_markdown_fence_region",
    "_target_function_present_for_eval",
    "format_success",
    "parse_success",
    "parse_success_for_eval",
    "postprocess_completion_for_eval",
]
