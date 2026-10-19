"""Third-party component inventory for commercial distribution.

One source of truth for *what ships*, *under what license*, and *what notice
is required*. The release tooling renders ``THIRD_PARTY_NOTICES.md`` from this
module and the commercial gate reads the same inventory, so the notice file
and the gate cannot disagree about a dependency.

Import discipline: stdlib plus ``packaging`` only. No network, no model
downloads, no torch/huggingface imports, and nothing runs at import time.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

#: The extras the managed installer actually installs. ``chatterbox`` is in the
#: project table for manual installs; the bootstrap installs the pinned
#: ``chatterbox-tts`` package separately, so both are part of the runtime story.
RUNTIME_EXTRAS: tuple[str, ...] = ("ml", "chatterbox")

#: Licenses that are compatible with a proprietary application as long as the
#: notice/attribution conditions are met.
PERMISSIVE = "permissive"
#: Weak copyleft: the library stays under its license, the application can stay
#: proprietary if the replacement/source obligations are met.
WEAK_COPYLEFT = "weak-copyleft"
#: Strong copyleft: a proprietary combined distribution is not safe without a
#: legal decision (replacement, separation, or a different license).
STRONG_COPYLEFT = "strong-copyleft"
PROPRIETARY = "proprietary"
UNKNOWN = "unknown"

_LICENSE_TEXT_FILES = {
    "LGPL-3.0": "LGPL-3.0.txt",
    "Apache-2.0": "Apache-2.0.txt",
    "MIT": "MIT.txt",
    "BSD-3-Clause": "BSD-3-Clause.txt",
    "GPL-2.0": "GPL-2.0.txt",
    "GPL-3.0": "GPL-3.0.txt",
    "MPL-2.0": "MPL-2.0.txt",
}


@dataclass(frozen=True, slots=True)
class Component:
    """One third-party component in (or optional to) the distribution."""

    name: str
    version: str
    license: str
    kind: str
    source: str = ""
    notice: str = ""

    @property
    def classification(self) -> str:
        return classify_license(self.license)


@dataclass(frozen=True, slots=True)
class Inventory:
    """The complete component set for one release tree."""

    components: tuple[Component, ...]
    version: str
    problems: tuple[str, ...] = field(default_factory=tuple)

    def by_kind(self, kind: str) -> tuple[Component, ...]:
        return tuple(component for component in self.components if component.kind == kind)


def classify_license(text: str) -> str:
    """Classify a license string into one of the five distribution buckets.

    The classifier is deliberately conservative: a name it does not recognize
    is ``unknown``, which the commercial gate treats as a blocker. Matching is
    substring-based because package metadata is inconsistent (``GPL-2.0-or-
    later``, ``GNU GENERAL PUBLIC LICENSE Version 3``, ``LGPL``, ...).
    """
    normalized = (text or "").strip().lower()
    if not normalized or normalized in {"unknown", "unlicensed", "none"}:
        return UNKNOWN
    # Use the title region for the GPL family: the full GPL-3 body mentions
    # the Lesser GPL in its "How to Apply" section, and a whole-document scan
    # would classify every GPL as weak copyleft. Short SPDX identifiers are
    # inside this window too.
    head = normalized[:400]
    # Check weak copyleft before strong: "LGPL" contains "GPL".
    if any(token in head for token in ("lgpl", "lesser general public", "mozilla public", "mpl-2", "epl-", "cddl")):
        return WEAK_COPYLEFT
    if any(
        token in head
        for token in (
            "agpl",
            "affero",
            "gpl-3",
            "gplv3",
            "gpl-2",
            "gplv2",
            "gnu general public license",
            "general public license version",
        )
    ):
        return STRONG_COPYLEFT
    if any(
        token in normalized
        for token in (
            "mit",
            "bsd",
            "apache",
            "isc",
            "psf",
            "python software foundation",
            "unlicense",
            "zlib",
            "cc0",
            "boost",
            "wtfpl",
            "public domain",
            "creative commons",
            "cc by",
        )
    ):
        return PERMISSIVE
    if any(
        token in normalized
        for token in ("proprietary", "commercial", "eula", "nvidia", "owner-owned", "all rights reserved")
    ):
        return PROPRIETARY
    return UNKNOWN


_CLASSIFIER_LICENSES = {
    "MIT License": "MIT",
    "BSD License": "BSD-3-Clause",
    "Apache Software License": "Apache-2.0",
    "ISC License (ISCL)": "ISC",
    "Python Software Foundation License": "PSF-2.0",
    "GNU Library or Lesser General Public License (LGPL)": "LGPL-3.0",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}


def _license_from_text(text: str) -> str:
    """Best-effort canonical license name from a metadata license field.

    Only the title region is used for the GPL family: a full GPL-3 body
    mentions the Lesser GPL in its "How to Apply" section, and scanning the
    whole body would misclassify every GPL package as LGPL.
    """
    lowered = text.lower()
    head = lowered[:400]
    if "lesser general public license" in head or "lgpl" in head:
        return "LGPL-3.0"
    if "affero" in head or "agpl" in head:
        return "AGPL-3.0"
    if "gnu general public license" in head or head.strip().startswith("gpl"):
        if "version 3" in head or "v3" in head or "gpl-3" in head:
            return "GPL-3.0"
        if "version 2" in head or "v2" in head or "gpl-2" in head:
            return "GPL-2.0"
        return "GPL"
    if "mozilla public license" in lowered or "mpl-2" in lowered:
        return "MPL-2.0"
    if "apache" in lowered:
        return "Apache-2.0"
    if "mit license" in lowered or "permission is hereby granted, free of charge" in lowered:
        return "MIT"
    if "bsd" in lowered or "redistribution and use in source and binary forms" in lowered:
        return "BSD-3-Clause"
    if "isc license" in lowered:
        return "ISC"
    if "python software foundation" in lowered or "psf" in lowered:
        return "PSF-2.0"
    if "public domain" in lowered or "cc0" in lowered:
        return "CC0-1.0"
    return ""


def _license_from_metadata(meta: metadata.PackageMetadata) -> str:
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    # The classifier is the most reliable short identifier when the free-text
    # License field is a full license body (scipy, numpy, language-tool-python).
    for classifier in meta.get_all("Classifier") or []:
        if not classifier.startswith("License :: OSI Approved ::"):
            continue
        value = classifier.rsplit("::", 1)[-1].strip()
        if value in _CLASSIFIER_LICENSES:
            return _CLASSIFIER_LICENSES[value]
        canonical = _license_from_text(value)
        if canonical:
            return canonical
    declared = (meta.get("License") or "").strip()
    if declared and declared.upper() != "UNKNOWN":
        canonical = _license_from_text(declared)
        if canonical:
            return canonical
        first = next((line.strip() for line in declared.splitlines() if line.strip()), "")
        if first and len(first) <= 80:
            return first
    return ""


def _source_from_metadata(meta: metadata.PackageMetadata) -> str:
    homepage = (meta.get("Home-page") or "").strip()
    if homepage:
        return homepage
    for value in meta.get_all("Project-URL") or []:
        if "," in value:
            _label, url = value.split(",", 1)
            url = url.strip()
            if url:
                return url
    return (meta.get("Download-URL") or "").strip()


def _component_from_distribution(dist: metadata.Distribution) -> Component:
    meta = dist.metadata
    name = meta.get("Name") or "unknown"
    return Component(
        name=name,
        version=dist.version or "",
        license=_license_from_metadata(meta),
        kind="python",
        source=_source_from_metadata(meta),
    )


def installed_distributions() -> dict[str, metadata.Distribution]:
    """Installed distributions keyed by canonical name, one entry per name."""
    chosen: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = canonicalize_name(name)
        current = chosen.get(key)
        if current is None or _version_key(dist.version) > _version_key(current.version):
            chosen[key] = dist
    return chosen


def _version_key(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in re.split(r"[.+-]", value or ""):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts)


def _selected_requirements(pyproject: Path, extras: tuple[str, ...]) -> list[Requirement]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    raw: list[str] = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies") or {}
    for extra in extras:
        raw.extend(optional.get(extra) or [])
    requirements: list[Requirement] = []
    for value in raw:
        try:
            requirements.append(Requirement(value))
        except InvalidRequirement:
            continue
    return requirements


def _marker_applies(requirement: Requirement) -> bool:
    if requirement.marker is None:
        return True
    try:
        return requirement.marker.evaluate(default_environment())
    except Exception:
        # A malformed marker must not silently shrink the inventory.
        return True


def runtime_components(
    pyproject: Path,
    *,
    installations: dict[str, metadata.Distribution] | None = None,
    extras: tuple[str, ...] = RUNTIME_EXTRAS,
) -> list[Component]:
    """Resolve the runtime dependency closure from ``pyproject.toml``.

    The closure is computed from package metadata, not from a hard-coded list,
    so adding a dependency cannot silently omit it from the notices. Extras on
    transitive requirements are not resolved unless they are selected at the
    root; that matches how pip installs the project's declared extras.
    """
    installations = installations or installed_distributions()
    selected_extras = {canonicalize_name(extra) for extra in extras}
    queue: list[Requirement] = [req for req in _selected_requirements(pyproject, extras) if _marker_applies(req)]
    seen: set[str] = set()
    components: list[Component] = []

    while queue:
        requirement = queue.pop(0)
        name = canonicalize_name(requirement.name)
        if name in seen:
            continue
        seen.add(name)
        dist = installations.get(name)
        if dist is None:
            # Keep an explicit missing entry: an uninstalled declared dependency
            # is a release-environment problem the caller should see.
            components.append(
                Component(
                    name=requirement.name,
                    version="",
                    license=UNKNOWN,
                    kind="python",
                    notice="Declared in pyproject.toml but not installed in this environment.",
                )
            )
            continue
        components.append(_component_from_distribution(dist))
        for raw in dist.requires or []:
            try:
                transitive = Requirement(raw)
            except InvalidRequirement:
                continue
            if not _marker_applies(transitive):
                continue
            # A requirement guarded by an extra is only relevant when that
            # extra was selected at the root.
            marker_text = str(transitive.marker or "")
            extra_match = re.search(r'extra\s*==\s*["\']([^"\']+)["\']', marker_text)
            if extra_match and canonicalize_name(extra_match.group(1)) not in selected_extras:
                continue
            queue.append(transitive)

    return sorted(components, key=lambda component: canonicalize_name(component.name))


#: Model repositories and the exact license facts verified on 2026-10-19.
_MODEL_COMPONENTS: tuple[Component, ...] = (
    Component(
        name="ResembleAI/chatterbox",
        version="pinned (models/pins.py)",
        license="MIT",
        kind="model",
        source="https://huggingface.co/ResembleAI/chatterbox",
        notice="Copyright (c) 2025 Resemble AI. MIT license; commercial use permitted.",
    ),
    Component(
        name="ResembleAI/chatterbox-turbo",
        version="pinned (models/pins.py)",
        license="MIT",
        kind="model",
        source="https://huggingface.co/ResembleAI/chatterbox-turbo",
        notice="MIT license; commercial use permitted.",
    ),
    Component(
        name="SamLowe/roberta-base-go_emotions",
        version="pinned (models/pins.py)",
        license="MIT",
        kind="model",
        source="https://huggingface.co/SamLowe/roberta-base-go_emotions",
        notice="Copyright (c) 2023 SamLowe. MIT license.",
    ),
    Component(
        name="oliverguhr/fullstop-punctuation-multilang-large",
        version="pinned (models/pins.py)",
        license="MIT",
        kind="model",
        source="https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large",
        notice="Copyright (c) 2021 Oliver Guhr. MIT license.",
    ),
)

#: External components that are not bundled in the wheel but are part of the
#: optional Vulkan path or the host runtime. They are listed so a reader is
#: never surprised by a component that the app invokes.
_EXTERNAL_COMPONENTS: tuple[Component, ...] = (
    Component(
        name="audio.cpp",
        version="built at install time from a pinned clone",
        license="Apache-2.0",
        kind="external",
        source="https://github.com/0xShug0/audio.cpp",
        notice="Copyright 2026 ShugoAI LLC. Apache-2.0. Modified by The Oracle's vendored patches; see scripts/patches/.",
    ),
    Component(
        name="ggml",
        version="vendored in the audio.cpp checkout",
        license="MIT",
        kind="external",
        source="https://github.com/ggml-org/ggml",
        notice="Copyright (c) 2023-2026 The ggml authors. MIT license.",
    ),
    Component(
        name="ffmpeg",
        version="host-provided",
        license="LGPL/GPL depending on the build",
        kind="external",
        source="https://ffmpeg.org/",
        notice="Not bundled. The host supplies ffmpeg; its own build determines the exact license.",
    ),
)

#: PySide6 and shiboken6 ship as native binaries; they are in the Python
#: closure too, but the LGPL obligation deserves its own entry.
_QT_COMPONENT = Component(
    name="PySide6 / shiboken6 / Qt",
    version="6.8.2.1",
    license="LGPL-3.0 (or commercial Qt license)",
    kind="bundled-binary",
    source="https://code.qt.io/cgit/pyside/pyside-setup.git/",
    notice=(
        "LGPL-3.0. A proprietary application may use the LGPL path while keeping "
        "its own source proprietary, provided the LGPL-covered libraries remain "
        "replaceable/relinkable, their complete corresponding source is made "
        "available (or a written offer is given), and the LGPL notice and text "
        "accompany the distribution. A Qt commercial license is the alternative."
    ),
)

_GENERIC_VOICE_LICENSES: dict[str, tuple[str, str]] = {
    "audio_cpp": ("Apache-2.0", "audio.cpp demo voice, Copyright 2026 ShugoAI LLC"),
    "librispeech": ("CC BY 4.0", "LibriSpeech clip; attribution in Seashells/generic/ATTRIBUTION.md"),
}


def _generic_voice_components(root: Path) -> list[Component]:
    directory = root / "Seashells" / "generic"
    if not directory.is_dir():
        return []
    components: list[Component] = []
    for path in sorted(directory.glob("*.wav")):
        if path.name.startswith(("english_male", "chinese_")):
            license_name, notice = _GENERIC_VOICE_LICENSES["audio_cpp"]
        else:
            license_name, notice = _GENERIC_VOICE_LICENSES["librispeech"]
        components.append(
            Component(
                name=path.name,
                version="",
                license=license_name,
                kind="voice",
                source="Seashells/generic/ATTRIBUTION.md",
                notice=notice,
            )
        )
    return components


def _owner_voice_components(root: Path) -> list[Component]:
    directory = root / "Seashells"
    if not directory.is_dir():
        return []
    components: list[Component] = []
    for path in sorted(directory.glob("*.wav")):
        components.append(
            Component(
                name=path.name,
                version="",
                license="Owner-owned; all rights reserved",
                kind="voice",
                source="Seashells/ASSETS.md",
                notice="Original recording by the Owner; no third-party rights.",
            )
        )
    return components


def bundled_components(root: Path) -> list[Component]:
    """Everything the repository ships or invokes beyond the Python closure."""
    components: list[Component] = [
        _QT_COMPONENT,
        *_MODEL_COMPONENTS,
        *_EXTERNAL_COMPONENTS,
        *_generic_voice_components(root),
        *_owner_voice_components(root),
    ]
    return components


def build_inventory(root: Path, *, version: str) -> Inventory:
    """Build the complete inventory for the tree at ``root``."""
    pyproject = root / "pyproject.toml"
    problems: list[str] = []
    if not pyproject.is_file():
        problems.append("pyproject.toml is missing; the runtime closure cannot be resolved")
        runtime: list[Component] = []
    else:
        runtime = runtime_components(pyproject)
    components = tuple(runtime) + tuple(bundled_components(root))
    return Inventory(components=components, version=version, problems=tuple(problems))


def _table_row(component: Component) -> str:
    source = component.source or ""
    if source.startswith("http"):
        source = f"[source]({source})"
    return f"| `{component.name}` | {component.version or '—'} | {component.license or 'UNKNOWN'} | {source or '—'} |"


def _required_notice(component: Component) -> str:
    return f"- **{component.name}** — {component.notice}" if component.notice else ""


def render_notices(inventory: Inventory) -> str:
    """Render the committed ``THIRD_PARTY_NOTICES.md`` deterministically."""
    python_components = sorted(
        (component for component in inventory.components if component.kind == "python"),
        key=lambda component: canonicalize_name(component.name),
    )
    non_python = sorted(
        (component for component in inventory.components if component.kind != "python"),
        key=lambda component: (component.kind, component.name.lower()),
    )

    lines: list[str] = [
        "# Third-Party Notices",
        "",
        "This file is generated from the release tree by",
        "`python scripts/generate_third_party_notices.py`. Do not edit it by hand;",
        "regenerate it and review the diff. The inventory is also read by the",
        "commercial launch gate (`python scripts/soft_launch.py --check`), so a",
        "dependency that is missing here is a release blocker, not a documentation",
        "omission.",
        "",
        f"Release version: **{inventory.version}**",
        "",
        "## Distribution obligations at a glance",
        "",
        "- **MIT / BSD / Apache-2.0 components** require their copyright and",
        "  permission notices to accompany copies. Their license texts are in",
        "  `THIRD_PARTY_LICENSES/` and in each Python distribution's `.dist-info`",
        "  directory.",
        "- **PySide6 / Qt (LGPL-3.0)** requires a prominent notice, a copy of the",
        "  LGPL text, access to the corresponding source of the LGPL-covered",
        "  libraries, and a way for users to replace/relink those libraries. A",
        "  commercial Qt license is the alternative.",
        "- **Strong-copyleft components (GPL/AGPL)** are not safe for a proprietary",
        "  combined distribution without a recorded legal decision. The commercial",
        "  gate reports them as blockers until `scripts/license_review.json` records",
        "  the decision.",
        "- **External tools** (ffmpeg, audio.cpp/ggml) are not part of the Python",
        "  wheel. The host supplies ffmpeg; the optional Vulkan backend builds",
        "  audio.cpp/ggml locally.",
        "",
        "## Bundled assets, models, and external components",
        "",
        "| Component | Version / pin | License | Source |",
        "|---|---|---|---|",
    ]
    lines.extend(_table_row(component) for component in non_python)
    lines.extend(
        [
            "",
            "Model revisions are pinned in `src/the_oracle/models/pins.py`; this",
            "table names each model and its license without duplicating the",
            "revision hashes, which are external identifiers rather than citations",
            "in this repository's history.",
            "",
            "## Python dependencies",
            "",
            "Resolved from `pyproject.toml` and the installed package metadata. The",
            "closure covers the runtime and the `ml`/`chatterbox` extras; development",
            "tools are not part of the shipped environment.",
            "",
            "| Component | Version | License | Source |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(_table_row(component) for component in python_components)
    lines.extend(
        [
            "",
            "## Required notices",
            "",
            "The following notices must accompany the distribution. The standard",
            "license texts are in `THIRD_PARTY_LICENSES/`.",
            "",
        ]
    )
    notices = [_required_notice(component) for component in non_python if component.notice]
    notices.extend(
        _required_notice(component)
        for component in python_components
        if component.classification in {STRONG_COPYLEFT, WEAK_COPYLEFT, UNKNOWN}
    )
    lines.extend(notices or ["- No additional notices are required beyond the license texts."])
    lines.extend(
        [
            "",
            "## Standard license texts",
            "",
        ]
    )
    for name, filename in sorted(_LICENSE_TEXT_FILES.items()):
        lines.append(f"- `{name}` — `THIRD_PARTY_LICENSES/{filename}`")
    lines.extend(
        [
            "",
            "## Full dependency license texts",
            "",
            "Every installed Python distribution carries its own license files in its",
            "`.dist-info` directory. The offline bundle ships the wheels, so those",
            "texts travel with the install. The standard texts above cover the",
            "components whose wheels do not carry a text (notably PySide6/Qt).",
            "",
        ]
    )
    return "\n".join(lines)


def notice_problems(root: Path, *, inventory: Inventory | None = None) -> list[str]:
    """Every way the committed notices disagree with the release tree."""
    path = root / "THIRD_PARTY_NOTICES.md"
    if not path.is_file():
        return ["THIRD_PARTY_NOTICES.md is missing; run scripts/generate_third_party_notices.py"]
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    problems: list[str] = []
    for forbidden in ("draft", "todo", "tbd", "undocumented"):
        if forbidden in lowered:
            problems.append(
                f"THIRD_PARTY_NOTICES.md contains {forbidden!r}; generated notices must be final"
            )
    if "soft_launch.py" not in text:
        problems.append("THIRD_PARTY_NOTICES.md does not name the commercial launch gate")
    if inventory is None:
        return problems
    for component in inventory.components:
        if f"`{component.name}`" not in text and component.name not in text:
            problems.append(f"THIRD_PARTY_NOTICES.md is missing component {component.name!r}")
    for name, filename in _LICENSE_TEXT_FILES.items():
        if not (root / "THIRD_PARTY_LICENSES" / filename).is_file():
            problems.append(f"THIRD_PARTY_LICENSES/{filename} is missing for {name}")
    return problems
