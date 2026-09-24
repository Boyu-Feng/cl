"""
==============================================================================
playbook.py
==============================================================================

This file contains functions for parsing and manipulating the playbook.

"""

import json
import re
from typing import Optional

from .utils import get_section_slug


def parse_playbook_line(line):
    """Parse a single playbook line to extract components"""
    # Pattern: [id] helpful=X harmful=Y :: content
    pattern = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern, line.strip())

    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
            "raw_line": line,
        }
    return None


def normalize_section_name(section_name):
    """Normalize playbook section names consistently across curator and parser."""
    return (
        section_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("&", "and")
    )


def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split("\n")

    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r"-(\d+)$", parsed["id"])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)

    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format"""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def update_bullet_counts(playbook_text, bullet_tags):
    """Update helpful/harmful counts based on tags (Counter layer)"""
    lines = playbook_text.strip().split("\n")
    updated_lines = []

    # Create tag lookup - handle both old and new formats
    tag_map = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Handle both 'id' and 'bullet' keys for backwards compatibility
                bullet_id = tag.get("id") or tag.get("bullet", "")
                tag_value = tag.get("tag", "neutral")
                if bullet_id:
                    tag_map[bullet_id] = tag_value

    if not tag_map:
        print("Warning: No valid bullet tags found to update counts")
        return playbook_text

    for line in lines:
        if line.strip().startswith("#") or not line.strip():
            # Preserve section headers and empty lines
            updated_lines.append(line)
            continue

        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in tag_map:
            tag = tag_map[parsed["id"]]
            if tag == "helpful":
                parsed["helpful"] += 1
            elif tag == "harmful":
                parsed["harmful"] += 1
            # neutral: no change

            # Reconstruct line with updated counts
            new_line = format_playbook_line(
                parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]
            )
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    return "\n".join(updated_lines)


def apply_curator_operations(playbook_text, operations, next_id):
    """Apply curator operations (ADD / UPDATE / MERGE / DELETE) to the playbook.

    - ADD: append a new bullet (helpful=0 harmful=0) to the named section.
    - UPDATE: rewrite an existing bullet's content, preserving its counters.
    - DELETE: remove an existing bullet outright.
    - MERGE: delete several source bullets and add one new bullet whose counters
      are the summed helpful/harmful of the sources.
    """
    lines = playbook_text.strip().split("\n")

    delete_ids: set[str] = set()
    update_map: dict[str, str] = {}
    pending_adds: list[
        tuple[str, int, int, str]
    ] = []  # (section, helpful, harmful, content)

    def _resolve_bullet_id(op):
        return op.get("bullet_id") or op.get("id")

    for op in operations:
        if not isinstance(op, dict):
            print(f"Warning: skipping non-dict operation: {op!r}")
            continue
        op_type = op.get("type")

        if op_type == "ADD":
            section = normalize_section_name(op.get("section", "general"))
            content = op.get("content", "")
            if not content:
                print(f"Warning: ADD operation missing content: {op}")
                continue
            pending_adds.append((section, 0, 0, content))

        elif op_type == "UPDATE":
            bullet_id = _resolve_bullet_id(op)
            new_content = op.get("content")
            if not bullet_id or not new_content:
                print(f"Warning: UPDATE operation missing bullet_id or content: {op}")
                continue
            update_map[bullet_id] = new_content

        elif op_type == "DELETE":
            bullet_id = _resolve_bullet_id(op)
            if not bullet_id:
                print(f"Warning: DELETE operation missing bullet_id: {op}")
                continue
            delete_ids.add(bullet_id)

        elif op_type == "MERGE":
            source_ids = op.get("source_ids") or op.get("bullet_ids") or []
            section = normalize_section_name(op.get("section", "general"))
            content = op.get("content", "")
            if not source_ids or not content:
                print(f"Warning: MERGE operation missing source_ids or content: {op}")
                continue
            helpful_sum = 0
            harmful_sum = 0
            for line in lines:
                parsed = parse_playbook_line(line)
                if parsed and parsed["id"] in source_ids:
                    helpful_sum += parsed["helpful"]
                    harmful_sum += parsed["harmful"]
            delete_ids.update(source_ids)
            pending_adds.append((section, helpful_sum, harmful_sum, content))

        else:
            print(f"Warning: unsupported operation type {op_type!r}; skipping")

    # Pass 1: apply DELETE + UPDATE in place over existing lines.
    surviving_lines: list[str] = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            if parsed["id"] in delete_ids:
                print(f"  Deleted bullet {parsed['id']}")
                continue
            if parsed["id"] in update_map:
                surviving_lines.append(
                    format_playbook_line(
                        parsed["id"],
                        parsed["helpful"],
                        parsed["harmful"],
                        update_map[parsed["id"]],
                    )
                )
                print(f"  Updated bullet {parsed['id']}")
                continue
        surviving_lines.append(line)

    # Discover which sections still exist (so ADDs can be routed sanely).
    sections_present: set[str] = set()
    for line in surviving_lines:
        if line.strip().startswith("##"):
            sections_present.add(normalize_section_name(line.strip()[2:].strip()))

    # Materialize pending ADDs (from ADD and MERGE) into formatted lines with fresh IDs.
    bullets_to_add: list[tuple[str, str]] = []
    for section, helpful, harmful, content in pending_adds:
        if section not in sections_present and section != "general":
            print(f"Warning: Section '{section}' not found, adding to OTHERS")
            section = "others"
        slug = get_section_slug(section)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        bullets_to_add.append(
            (section, format_playbook_line(new_id, helpful, harmful, content))
        )
        print(f"  Added bullet {new_id} to section {section}")

    # Pass 2: walk surviving lines and append new bullets at the tail of each section.
    final_lines: list[str] = []
    current_section: Optional[str] = None
    for line in surviving_lines:
        if line.strip().startswith("##"):
            if current_section:
                section_adds = [b for s, b in bullets_to_add if s == current_section]
                final_lines.extend(section_adds)
                bullets_to_add = [
                    (s, b) for s, b in bullets_to_add if s != current_section
                ]
            current_section = normalize_section_name(line.strip()[2:].strip())
        final_lines.append(line)

    if current_section:
        section_adds = [b for s, b in bullets_to_add if s == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    # Anything still unplaced (section never existed) lands in OTHERS.
    if bullets_to_add:
        print(
            f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS"
        )
        leftover = [b for _, b in bullets_to_add]
        others_idx = -1
        for i, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_idx = i
                break
        if others_idx >= 0:
            for i, bullet in enumerate(leftover):
                final_lines.insert(others_idx + 1 + i, bullet)
        else:
            final_lines.extend(leftover)

    return "\n".join(final_lines), next_id


def get_playbook_stats(playbook_text):
    """Generate statistics about the playbook"""
    lines = playbook_text.strip().split("\n")
    stats = {
        "total_bullets": 0,
        "high_performing": 0,  # helpful > 5, harmful < 2
        "problematic": 0,  # harmful >= helpful
        "unused": 0,  # helpful + harmful = 0
        "by_section": {},
    }

    current_section = "general"

    for line in lines:
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue

        parsed = parse_playbook_line(line)
        if parsed:
            stats["total_bullets"] += 1

            if parsed["helpful"] > 5 and parsed["harmful"] < 2:
                stats["high_performing"] += 1
            elif parsed["harmful"] >= parsed["helpful"] and parsed["harmful"] > 0:
                stats["problematic"] += 1
            elif parsed["helpful"] + parsed["harmful"] == 0:
                stats["unused"] += 1

            if current_section not in stats["by_section"]:
                stats["by_section"][current_section] = {
                    "count": 0,
                    "helpful": 0,
                    "harmful": 0,
                }

            stats["by_section"][current_section]["count"] += 1
            stats["by_section"][current_section]["helpful"] += parsed["helpful"]
            stats["by_section"][current_section]["harmful"] += parsed["harmful"]

    return stats


def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats"""
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass

        # Fallback: Look for ```json blocks
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue

        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == "{":
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1

                    while i < len(text) and brace_count > 0:
                        if text[i] == "{":
                            brace_count += 1
                        elif text[i] == "}":
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == "\\":
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1

                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1

            return json_objects

        # Find all potential JSON objects
        json_objects = find_json_objects(text)

        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if len(text) > 500:
            print(f"Raw content preview:\n{text[:500]}...")
        else:
            print(f"Raw content:\n{text}")

    return None


def extract_playbook_bullets(playbook_text, bullet_ids):
    """
    Extract specific bullet points from playbook based on bullet_ids.

    Args:
        playbook_text (str): The full playbook text
        bullet_ids (list): List of bullet IDs to extract

    Returns:
        str: Formatted playbook content containing only the specified bullets
    """
    if not bullet_ids:
        return "(No bullets used by generator)"

    lines = playbook_text.strip().split("\n")
    found_bullets = []

    for line in lines:
        if line.strip():  # Skip empty lines
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] in bullet_ids:
                found_bullets.append(
                    {
                        "id": parsed["id"],
                        "content": parsed["content"],
                        "helpful": parsed["helpful"],
                        "harmful": parsed["harmful"],
                    }
                )

    if not found_bullets:
        return "(Generator referenced bullet IDs but none were found in playbook)"

    # Format the bullets for reflector input
    formatted_bullets = []
    for bullet in found_bullets:
        formatted_bullets.append(
            f"[{bullet['id']}] helpful={bullet['helpful']} harmful={bullet['harmful']} :: {bullet['content']}"
        )

    return "\n".join(formatted_bullets)
