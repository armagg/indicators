# GitHub automation

The changes in this pull request become the default automation after merging. CI runs on pull requests, pushes to `main`, manual dispatch, and Mondays at 04:12 UTC. Scheduled and manual runs require the workflow to exist on the default branch.

The workflow runs the Python test suite and builds the Docker image after tests pass. The normalization fix treats almost constant spreads as neutral while preserving real variation and missing data.

Actions are pinned to inspected commit SHAs, repository access is read-only, checkouts do not retain credentials, jobs have time limits, and a newer run cancels an older run for the same branch. Docker builds do not log in, publish images, or deploy services.

Dependabot proposes weekly dependency/Action updates. Review and test each proposal before merging. Major upgrades and changes to numerical libraries need compatibility testing. Dependabot version updates require the configuration on the default branch. Security-update settings are separate repository settings.

## Deployment and ongoing operation

Use the existing VPS/container infrastructure for continuous bot operation. GitHub-hosted jobs have execution limits and scheduled jobs can be delayed. CI prepares and checks a change; it does not keep a bot running continuously.

Before adding a publish/deploy workflow, confirm the actual host, SSH identity, registry access, compose project/service names, current rollback/active-trial guards, and retained image digests. Only trusted default-branch/tag builds should publish. Deploy the exact tested digest through the existing scoped deployment and rollback controls. Production environment reviewers and required checks must be configured in repository settings where the plan supports them; writing `environment: production` alone does not enforce approval.

Release notes, draft releases, container publication, smoke checks, fleet monitoring, and backup/restore exercises can be added once their inputs and credentials are configured. None of those production operations is enabled by this PR.

## Weekly dependency advisory checks

The separate Dependency security workflow runs manually and on Wednesdays at 05:12 UTC. Go uses govulncheck v1.9.0; Python uses pip-audit 2.10.1 against the public dependencies resolved in a separate runtime environment. The local SDK itself is excluded from PyPI submission; its own repository audits its public dependencies. Packaging tools (pip, setuptools, wheel) are excluded from this application dependency check. No findings are automatically fixed or ignored. These scans can fail on existing vulnerabilities until the pending security updates are merged. They do not scan container OS packages or application logic.
