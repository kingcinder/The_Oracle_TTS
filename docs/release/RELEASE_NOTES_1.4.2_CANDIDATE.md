# The Oracle TTS 1.4.2 Candidate — Release Notes

This engineering candidate preserves Community workflows without activation,
keeps `render-report` behind its existing Studio entitlement, limits the
managed Chatterbox NumPy dependency to the compatible major version, and adds
an `ml-dtypes` constraint for clean dependency resolution. It also improves
peak headroom and exercises a two-speaker real-engine smoke with spoken
references. The support-report workflows and third-party notices are included.

The candidate is **not approved for commercial release**. Linux x86_64 has
passed an isolated managed install, doctor, dependency check, and one real CPU
render, but the complete first-user/project workflow, candidate hosted CI,
manual intelligibility review, and release artifact verification remain open.
The EULA is still a draft. See [Final Release Readiness](FINAL_RELEASE_READINESS.md)
and [Known Issues](KNOWN_ISSUES.md).
