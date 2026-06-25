# Security Policy

## Reporting a vulnerability

Please report security issues **privately** using GitHub's "Report a vulnerability"
button on the repository's **Security** tab (private vulnerability reporting is
enabled). Do not open a public issue for a suspected vulnerability.

We aim to acknowledge reports within a few days.

## Scope

PocketLM is an educational project: a tiny language model you train yourself in
notebooks. It ships no network services, no authentication, and handles no user
data. The realistic risk surface is the **software supply chain** (Python and
GitHub Actions dependencies), which is monitored by Dependabot and GitHub secret
scanning.

Responsible disclosure of anything you find, including in the example code or
notebooks, is appreciated.
